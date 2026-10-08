from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from lib.config.cgtptrack.config import cfg, update_config_from_file
from lib.models.cgtptrack import build_cgtptrack
from lib.models.cgtptrack.vit import Attention_Embx, Block_Embx

ROOT = Path(__file__).resolve().parents[1]


def test_attention_matches_separate_softmax_equation_and_ignores_padding():
    torch.manual_seed(7)
    attention = Attention_Embx(4, num_heads=1, qkv_bias=False).eval()
    query = torch.randn(1, 2, 4)
    keys = torch.randn(1, 5, 4)
    valid = torch.tensor([[True, True, False]])
    result, weights = attention(keys, query, True, memory_lambda=0.35, memory_mask=valid)
    q, k, v = attention.q(query), attention.k(keys), attention.v(keys)
    logits = q @ k.transpose(-1, -2) / 2
    a_search = logits[:, :, :2].softmax(-1)
    a_memory = logits[:, :, 2:4].softmax(-1)
    expected = attention.proj(.65 * (a_search @ v[:, :2]) + .35 * (a_memory @ v[:, 2:4]))
    torch.testing.assert_close(result, expected)
    assert weights[..., -1].eq(0).all()
    # Memory maintenance consumes independently normalized branch probabilities,
    # not mixing-coefficient-scaled weights (also valid when lambda_m=0).
    torch.testing.assert_close(weights[..., 2:].sum(-1), torch.ones(1, 1, 2))
    changed = keys.clone()
    changed[:, -1] = 10000
    torch.testing.assert_close(attention(changed, query, memory_mask=valid), result)


def test_role_specific_normalization_shared_source_projections_and_read_only_memory():
    block = Block_Embx(8, 2).eval()
    query, memory = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    original = memory.clone()
    out = block(torch.cat([query, memory], 1), query)
    assert out.shape == query.shape
    assert block.norm1 is not block.norm1_kv
    assert not hasattr(block.attn, 'k_mem') and not hasattr(block.attn, 'v_mem')
    assert torch.equal(original, memory)


@pytest.mark.parametrize('name', ['baseline', 'baseline_full'])
def test_paper_configs_and_registered_keys(name):
    experiment = yaml.safe_load((ROOT / 'experiments/cgtptrack' / (name + '.yaml')).read_text())
    local = deepcopy(cfg)
    update_config_from_file(str(ROOT / 'experiments/cgtptrack' / (name + '.yaml')), base_cfg=local)
    assert local.TRAIN.EPOCH == 300 and local.TRAIN.BATCH_SIZE == 8
    assert local.DATA.SEARCH.NUMBER == 4
    assert local.DATA.MAX_SAMPLE_INTERVAL == 200
    assert local.DATA.TRAIN.SAMPLE_PER_EPOCH == 15000
    assert local.DATA.TRAIN.DATASETS_RATIO == [1] * len(local.DATA.TRAIN.DATASETS_NAME)
    assert set(local.MODEL) == {'PRETRAIN_FILE', 'BACKBONE', 'CTP', 'HEAD'}
    assert set(local.MODEL.BACKBONE) == {'TYPE', 'STRIDE', 'MEMORY_LAMBDA'}
    assert set(local.MODEL.CTP) == {'ALPHA', 'TAU', 'CONF_THR', 'MEMORY_CONF_THR', 'SHORT_BETA', 'LONG_BETA', 'LONG_THR'}
    if name == 'baseline':
        assert local.DATA.TRAIN.DATASETS_NAME == ['GOT10K_train_full']
        assert local.DATA.VAL.DATASETS_NAME == ['GOT10K_official_val']


def test_full_model_forward_backward_and_checkpoint_namespace():
    # Real ViT-Base, reduced spatial size for a CPU smoke test; no pretrained weights.
    torch.set_num_threads(2)
    local = deepcopy(cfg)
    local.MODEL.PRETRAIN_FILE = ''
    local.DATA.SEARCH.SIZE = local.DATA.TEMPLATE.SIZE = 32
    local.MODEL.HEAD.NUM_CHANNELS = 16
    local.MODEL.CTP.CONF_THR = local.MODEL.CTP.LONG_THR = 0.
    local.MODEL.CTP.MEMORY_CONF_THR = 0.
    model = build_cgtptrack(local, training=False).train()
    result = model(torch.randn(2, 3, 32, 32), torch.randn(2, 2, 3, 32, 32))
    assert model.__class__.__name__ == 'CGTPTrack'
    assert result['pred_boxes'].shape == (4, 1, 4)
    assert result['prompt_state'].shape == (2, 2, 768)
    assert torch.isfinite(result['score_map']).all()
    (result['score_map'].sum() + result['pred_boxes'].sum()).backward()
    assert model.ctp.proj_s.weight.grad is not None
    assert torch.isfinite(model.ctp.proj_s.weight.grad).all()
    assert model.backbone.blocks[0].norm1_kv.weight.grad is not None


def test_new_role_norm_has_new_parameter_learning_rate():
    from lib.train.base_functions import get_optimizer_scheduler
    net = torch.nn.Module()
    net.backbone = torch.nn.Module()
    net.backbone.norm1 = torch.nn.LayerNorm(4)
    net.backbone.norm1_kv = torch.nn.LayerNorm(4)
    net.head = torch.nn.Linear(4, 1)
    net.head.bias.requires_grad_(False)
    optimizer, _ = get_optimizer_scheduler(net, cfg)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert rates[id(net.backbone.norm1.weight)] == cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER
    assert rates[id(net.backbone.norm1_kv.weight)] == cfg.TRAIN.LR
    assert id(net.head.bias) not in rates


def test_paper_resolution_forward():
    torch.set_num_threads(2)
    local = deepcopy(cfg)
    update_config_from_file(str(ROOT / 'experiments/cgtptrack/baseline_full.yaml'), base_cfg=local)
    local.MODEL.PRETRAIN_FILE = ''
    model = build_cgtptrack(local, training=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 256, 256), torch.randn(1, 1, 3, 256, 256))
    assert output['score_map'].shape == (1, 1, 16, 16)
    assert output['pred_boxes'].shape == (1, 1, 4)
    assert torch.isfinite(output['score_map']).all()
