"""Exercise the real online path on CPU; only CUDA placement and disk I/O are bypassed."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from lib.config.cgtptrack.config import cfg
from lib.models.cgtptrack import build_cgtptrack


def test_online_tracker_initialization_and_scheduled_memory(monkeypatch):
    from lib.test.tracker import cgtptrack as tracker_module

    torch.set_num_threads(2)
    local = deepcopy(cfg)
    local.MODEL.PRETRAIN_FILE = ''
    local.DATA.SEARCH.SIZE = local.DATA.TEMPLATE.SIZE = 32
    local.TEST.SEARCH_SIZE = local.TEST.TEMPLATE_SIZE = 32
    local.TEST.MEM_UPDATE_INTERVAL = 2
    local.MODEL.HEAD.NUM_CHANNELS = 16
    local.MODEL.CTP.MEMORY_CONF_THR = 0.
    net = build_cgtptrack(local, training=False)
    # Keep actual network, tracker, preprocessing, CTP and box decoding intact.
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(tracker_module, 'build_cgtptrack', lambda *a, **k: net)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {'net': net.state_dict()})
    params = SimpleNamespace(cfg=local, checkpoint='in_memory_test', debug=0,
                             save_all_boxes=False, template_factor=4.,
                             search_factor=4., template_size=32, search_size=32)
    tracker = tracker_module.CGTPTrack(params, 'lasot')
    image = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    tracker.initialize(image, {'init_bbox': [40., 40., 24., 24.]})
    initial = tracker.z.clone()
    first = tracker.track(image)
    assert torch.equal(tracker.z, initial)
    assert len(first['target_bbox']) == 4
    second = tracker.track(image)
    assert tracker.z.shape[1] == 8
    assert tracker.last_memory_update == 2
    assert np.isfinite(second['target_bbox']).all()
    # No stale reliable frame may authorize writing on the next scheduled frame.
    tracker.network.memory_conf_thr = 1.
    stable = tracker.z.clone()
    tracker.track(image)
    tracker.track(image)
    assert torch.equal(tracker.z, stable)
    tracker.initialize(image, {'init_bbox': [40., 40., 24., 24.]})
    assert tracker.z.shape[1] == 4 and tracker.tcm_state is None
