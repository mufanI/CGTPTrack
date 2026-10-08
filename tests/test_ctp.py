"""Numerical regression tests independent of datasets and trained weights."""
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'lib/models/cgtptrack/confidence_temporal_prompt.py'
spec = importlib.util.spec_from_file_location('ctp_under_test', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ConfidenceTemporalPrompt = module.ConfidenceTemporalPrompt


def test_closed_prompt_gate_preserves_initial_state_exactly():
    ctp = ConfidenceTemporalPrompt(4, conf_thr=1.0, long_thr=1.0)
    previous = ctp.init_state(2)
    updated, _, _, _ = ctp(torch.randn(2, 4, 4), torch.zeros(2, 4), previous)
    assert torch.equal(updated, previous)


@pytest.mark.parametrize('interval,frame,confidence', [(1, 1, 0.1), (3, 3, 0.1), (3, 2, 0.9)])
def test_memory_unchanged_without_current_write_permission(interval, frame, confidence):
    ctp = ConfidenceTemporalPrompt(4, conf_thr=0.5)
    memory = torch.randn(1, 8, 4)
    updated, _, committed = ctp.update_tcm_memory(
        memory, torch.rand(1, 4, 12), torch.rand(1, 4), 4, 4,
        torch.randn(1, 4, 4), frame_id=frame, commit_interval=interval,
        conf=torch.tensor([confidence]))
    assert not committed
    assert torch.equal(updated, memory)


def test_memory_uses_current_frame_and_keeps_top_ranked_history():
    ctp = ConfidenceTemporalPrompt(2, conf_thr=0.5)
    memory = torch.arange(16.).view(1, 8, 2)
    attention = torch.zeros(1, 4, 12)
    attention[:, :, 8:] = torch.arange(1., 5.)
    current = torch.full((1, 4, 2), 99.)
    updated, _, committed = ctp.update_tcm_memory(
        memory, attention, torch.ones(1, 4), 4, 4, current,
        frame_id=3, commit_interval=3, conf=torch.tensor([0.9]))
    assert committed
    assert torch.equal(updated[:, :4], memory[:, [7, 6, 5, 4]])
    assert torch.equal(updated[:, 4:], current)


def test_token_importance_matches_paper_equation():
    raw = torch.tensor([[0.1, 0.8]])
    refined = torch.tensor([[0.2, 0.9]])
    bias = torch.tensor([[-1., 1.]])
    result = ConfidenceTemporalPrompt.token_importance(raw, refined, bias, torch.tensor([0.75]))
    torch.testing.assert_close(result, torch.tensor([[0.025, 0.875]]))


def test_reliability_and_refinement_order():
    ctp = ConfidenceTemporalPrompt(4)
    logits = torch.tensor([[-2., 0., 1., 3.]])
    _, refined, bias, reliability = ctp(torch.randn(1, 4, 4), logits, None)
    p = (logits + ctp.alpha * bias).sigmoid()
    a = p / p.sum(1, keepdim=True)
    entropy = -(a * a.log()).sum(1) / torch.log(torch.tensor(4.))
    top = p.topk(2, dim=1).values
    expected = .45 * p.max(1).values + .35 * (1 - entropy) + .20 * (top[:, 0] - top[:, 1])
    torch.testing.assert_close(reliability, expected)
    torch.testing.assert_close(refined, logits + ctp.alpha * reliability[:, None] * bias)


def test_mixed_batch_gates_preserve_rejected_samples_and_gradients():
    ctp = ConfidenceTemporalPrompt(2, conf_thr=0.5)
    memory = torch.randn(2, 4, 2, requires_grad=True)
    current = torch.randn(2, 4, 2, requires_grad=True)
    updated, state, committed = ctp.update_tcm_memory(
        memory, torch.rand(2, 4, 8), torch.ones(2, 4), 4, 4, current,
        conf=torch.tensor([0.9, 0.1]))
    assert committed
    assert torch.equal(updated[1, :4], memory[1])
    assert state['valid_mask'][1].tolist() == [True] * 4 + [False] * 4
    assert state['valid_mask'][0].all()
    updated.sum().backward()
    assert current.grad[0].abs().sum() > 0
    assert current.grad[1].abs().sum() == 0
