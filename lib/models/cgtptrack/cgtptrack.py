"""CGTPTrack: BDM-ViT, localization and reliability-guided temporal state."""
from pathlib import Path

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224
from .confidence_temporal_prompt import ConfidenceTemporalPrompt


class CGTPTrack(nn.Module):
    def __init__(self, transformer, box_head, cfg):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.seg = nn.Embedding(2, transformer.embed_dim)
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = self.feat_sz_s ** 2
        self.memory_lambda = float(cfg.MODEL.BACKBONE.MEMORY_LAMBDA)
        ctp = cfg.MODEL.CTP
        self.memory_conf_thr = float(ctp.MEMORY_CONF_THR)
        self.ctp = ConfidenceTemporalPrompt(
            dim=transformer.embed_dim, alpha=ctp.ALPHA, tau=ctp.TAU,
            conf_thr=ctp.CONF_THR, short_beta=ctp.SHORT_BETA,
            long_beta=ctp.LONG_BETA, long_thr=ctp.LONG_THR)

    @staticmethod
    def _prob_to_logit(probability, eps=1e-6):
        probability = probability.clamp(eps, 1.0 - eps)
        return torch.log(probability / (1.0 - probability))

    def refine_response(self, search_tokens, raw_score_map, prompt_state):
        """Shared CTP path for training and online tracking."""
        prompt_state, logits, bias, reliability = self.ctp(
            search_tokens, self._prob_to_logit(raw_score_map), prompt_state)
        refined = logits.sigmoid()
        importance = self.ctp.token_importance(raw_score_map, refined, bias, reliability)
        return prompt_state, refined, importance, reliability

    def memory_candidates(self, search_tokens, importance):
        labels = (importance.flatten(1) > 0.5).long()
        return search_tokens + self.seg(labels)

    def forward(self, template, search, prompt_state=None):
        """template: (B,3,H,W), search: (T,B,3,H,W), state: (B,2,C)."""
        z = self.backbone.inference_template(template)
        if prompt_state is None:
            prompt_state = self.ctp.init_state(z.shape[0], device=z.device, dtype=z.dtype)
        memory_state = self.ctp.init_tcm_state(z.shape[0])
        boxes, responses = [], []
        for current in search:
            features, _, attention = self.backbone(
                x=current, z=z, memory_lambda=self.memory_lambda,
                memory_mask=memory_state['valid_mask'])
            tokens = features[:, -self.feat_len_s:]
            out = self.forward_head(tokens)
            prompt_state, refined, importance, reliability = self.refine_response(
                tokens, out['score_map'], prompt_state)
            pred_boxes = self.box_head.cal_bbox(refined, out['size_map'], out['offset_map'])
            z, memory_state, _ = self.ctp.update_tcm_memory(
                z, attention, importance, self.feat_len_s, self.feat_len_s,
                self.memory_candidates(tokens, importance), memory_state,
                commit_interval=1, conf=reliability, write_conf_thr=self.memory_conf_thr)
            boxes.append(pred_boxes.reshape(z.shape[0], 1, 4))
            responses.append(refined)
        return {'pred_boxes': torch.stack(boxes).flatten(0, 1),
                'score_map': torch.stack(responses).flatten(0, 1),
                'prompt_state': prompt_state}

    def forward_head(self, features):
        tokens = features[:, -self.feat_len_s:]
        spatial = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], self.feat_sz_s, self.feat_sz_s)
        score, bbox, size, offset = self.box_head(spatial, None)
        return {'pred_boxes': bbox.reshape(tokens.shape[0], 1, 4),
                'score_map': score, 'size_map': size, 'offset_map': offset}


def build_cgtptrack(cfg, training=True):
    if cfg.MODEL.BACKBONE.TYPE != 'vit_base_patch16_224' or cfg.MODEL.HEAD.TYPE != 'CENTER':
        raise ValueError('CGTPTrack uses the paper ViT-Base backbone and CENTER localization head.')
    checkpoint = cfg.MODEL.PRETRAIN_FILE
    full_model = 'CGTPTrack' in Path(checkpoint).name
    pretrained = ''
    if checkpoint and training and not full_model:
        pretrained = str(Path(__file__).resolve().parents[3] / 'pretrained_models' / checkpoint)
    backbone = vit_base_patch16_224(
        pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        memory_lambda=cfg.MODEL.BACKBONE.MEMORY_LAMBDA)
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    model = CGTPTrack(backbone, build_box_head(cfg, backbone.embed_dim), cfg)
    if full_model and training:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    return model
