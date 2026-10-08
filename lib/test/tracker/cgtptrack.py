"""Online CGTPTrack inference matching the manuscript's recurrent state loop."""
import torch

from lib.models.cgtptrack import build_cgtptrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class CGTPTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.cfg = params.cfg
        self.network = build_cgtptrack(self.cfg, training=False)
        checkpoint = torch.load(params.checkpoint, map_location='cpu')['net']
        self.network.load_state_dict(checkpoint, strict=True)
        self.network = self.network.cuda().eval()
        self.preprocessor = Preprocessor()
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_len_s = self.feat_sz ** 2
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]), centered=True).cuda()
        self.memory_lambda = float(self.cfg.MODEL.BACKBONE.MEMORY_LAMBDA)
        self.mem_update_interval = max(1, int(self.cfg.TEST.MEM_UPDATE_INTERVAL))
        self.save_all_boxes = params.save_all_boxes
        self.state = None
        self.prompt_state = None
        self.tcm_state = None
        self.frame_id = 0
        self.last_memory_update = 0

    @torch.no_grad()
    def initialize(self, image, info):
        patch, _, mask = sample_target(
            image, info['init_bbox'], self.params.template_factor,
            output_sz=self.params.template_size)
        template = self.preprocessor.process(patch, mask)
        self.z = self.network.backbone.inference_template(template.tensors)
        self.prompt_state = self.network.ctp.init_state(1, device=self.z.device, dtype=self.z.dtype)
        self.tcm_state = None
        self.state = list(info['init_bbox'])
        self.frame_id = 0
        self.last_memory_update = 0
        if self.save_all_boxes:
            return {'all_boxes': list(info['init_bbox'])}

    @torch.no_grad()
    def track(self, image, info=None):
        height, width = image.shape[:2]
        self.frame_id += 1
        patch, resize_factor, mask = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(patch, mask)
        features, _, attention = self.network.backbone(
            x=search.tensors, z=self.z, memory_lambda=self.memory_lambda)
        tokens = features[:, -self.feat_len_s:]
        out = self.network.forward_head(tokens)
        self.prompt_state, refined, importance, reliability = self.network.refine_response(
            tokens, out['score_map'], self.prompt_state)
        boxes = self.network.box_head.cal_bbox(
            self.output_window * refined, out['size_map'], out['offset_map'])
        box = (boxes.reshape(-1, 4).mean(0) * self.params.search_size / resize_factor).tolist()
        new_state = clip_box(self.map_box_back(box, resize_factor), height, width, margin=10)
        all_boxes = None
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                boxes * self.params.search_size / resize_factor, resize_factor).reshape(-1).tolist()
        self.z, self.tcm_state, committed = self.network.ctp.update_tcm_memory(
            self.z, attention, importance, self.feat_len_s, self.feat_len_s,
            self.network.memory_candidates(tokens, importance), self.tcm_state,
            frame_id=self.frame_id, commit_interval=self.mem_update_interval,
            conf=reliability, write_conf_thr=self.network.memory_conf_thr)
        if committed:
            self.last_memory_update = self.frame_id
        self.state = new_state
        result = {'target_bbox': self.state}
        if all_boxes is not None:
            result['all_boxes'] = all_boxes
        return result

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        enc_attn_weights = []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CGTPTrack
