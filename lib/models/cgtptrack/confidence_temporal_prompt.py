import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class ConfidenceTemporalPrompt(nn.Module):
    """Reliability-guided short/long prompts and scheduled memory maintenance.

    Response refinement and memory importance follow the manuscript's CTP
    equations. Prompt state has shape (B, 2, C), ordered short then long.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 0.5,
        tau: float = 0.07,
        conf_thr: float = 0.20,
        short_beta: float = 0.25,
        long_beta: float = 0.03,
        long_thr: float = 0.55,
    ):
        super().__init__()
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.conf_thr = float(conf_thr)
        self.short_beta = float(short_beta)
        self.long_beta = float(long_beta)
        self.long_thr = float(long_thr)

        # learnable initial prompts q0: short-term adapts fast, long-term updates slowly
        self.q0_short = nn.Parameter(torch.zeros(self.dim))
        self.q0_long = nn.Parameter(torch.zeros(self.dim))

        # projections for similarity bias
        self.proj_s = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_q = nn.Linear(self.dim, self.dim, bias=False)

        # GRU update
        self.gru = nn.GRUCell(self.dim, self.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q0_short, std=0.02)
        nn.init.normal_(self.q0_long, std=0.02)
        nn.init.xavier_uniform_(self.proj_s.weight)
        nn.init.xavier_uniform_(self.proj_q.weight)
        # GRU 默认初始化即可

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """
        返回 (B, 2, C) prompt 初始状态，index 0 是短期 prompt，index 1 是长期 prompt。
        """
        if device is None:
            device = self.q0_short.device
        if dtype is None:
            dtype = self.q0_short.dtype
        q_short = self.q0_short.unsqueeze(0).repeat(int(batch_size), 1)
        q_long = self.q0_long.unsqueeze(0).repeat(int(batch_size), 1)
        return torch.stack([q_short, q_long], dim=1).to(device=device, dtype=dtype)

    def get_prompt(self, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        将内部 prompt state 转成当前帧用于 attention/bias 的单个 prompt token。
        兼容旧调用里的 (B, C) state。
        """
        if state is None:
            return None
        if state.dim() == 2:
            return state
        if state.dim() == 3:
            q_short, q_long = state[:, 0], state[:, 1]
            return F.normalize(0.7 * q_short + 0.3 * q_long, dim=-1)
        raise ValueError(f"Unsupported prompt state shape: {tuple(state.shape)}")

    def _split_state(self, state: Optional[torch.Tensor], batch_size: int, device, dtype):
        if state is None:
            state = self.init_state(batch_size, device=device, dtype=dtype)
        if state.dim() == 2:
            state = torch.stack([state, state], dim=1)
        assert state.shape == (batch_size, 2, self.dim), f"prompt state must be (B,2,C), got {tuple(state.shape)}"
        return state[:, 0], state[:, 1]

    @staticmethod
    def _flatten_map(x: torch.Tensor) -> torch.Tensor:
        """
        将 score/map 统一展平到 (B, N)
        支持：
          - (B,1,H,W)
          - (B,H,W)
          - (B,N)
        """
        if x.dim() == 4:
            return x.flatten(2).squeeze(1)
        if x.dim() == 3:
            return x.flatten(1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unsupported map dim={x.dim()}, shape={tuple(x.shape)}")

    @staticmethod
    def token_importance(raw, refined, bias, reliability):
        """u = (1-r) C + r C_refined (1+b)/2, preserving response shape."""
        r = reliability.reshape(-1, *([1] * (raw.ndim - 1)))
        return (1.0 - r) * raw + r * refined * (1.0 + bias) * 0.5

    def init_tcm_state(self, batch_size: int, device=None, dtype=None) -> Dict[str, Any]:
        # Padding is needed only when independently gated batch members acquire
        # different memory lengths. False entries are excluded from attention.
        return {"valid_mask": None, "frame_count": 0}

    def update_tcm_memory(
        self, z, attn, score_map, skip_ref, feat_len_s, append_feat,
        tcm_state=None, frame_id=None, commit_interval=1, conf=None,
        write_conf_thr=None,
    ):
        """Write CURRENT tokens only on a scheduled, reliable frame.

        Keep TopK historical tokens ranked by current attention weighted by u.
        Return (memory, state, any_sample_committed). state['valid_mask'] must
        be passed to the backbone when a mixed batch requires zero padding.
        Unreliable samples preserve all existing valid tokens exactly.
        No interval averaging, best-candidate cache, or fallback writing occurs.
        """
        if tcm_state is None:
            tcm_state = self.init_tcm_state(z.shape[0])
        tcm_state['frame_count'] += 1
        frame = tcm_state['frame_count'] if frame_id is None else int(frame_id)
        if frame % max(1, int(commit_interval)) != 0 or conf is None:
            return z, tcm_state, False
        threshold = self.conf_thr if write_conf_thr is None else float(write_conf_thr)
        gate = conf.to(device=z.device).reshape(z.shape[0]) > threshold
        if not bool(gate.any()):
            return z, tcm_state, False

        importance = self._flatten_map(score_map).to(attn.dtype)
        history_attn = attn[:, :, int(skip_ref):]
        if history_attn.shape[-1] != z.shape[1]:
            raise ValueError('Historical attention length does not match memory.')
        valid = tcm_state.get('valid_mask')
        if valid is None:
            valid = torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)
        # Rank only samples whose write gate is open.
        scores = (history_attn[gate] * importance[gate, :, None]).sum(dim=1)
        scores = scores.masked_fill(~valid[gate], float('-inf'))
        keep = min(int(feat_len_s), z.shape[1])
        indices = scores.topk(keep, dim=1).indices
        retained = z[gate].gather(1, indices[..., None].expand(-1, -1, z.shape[2]))
        retained_valid = valid[gate].gather(1, indices)
        candidates = torch.cat([retained, append_feat[gate]], dim=1)
        candidate_valid = torch.cat([
            retained_valid,
            torch.ones(append_feat[gate].shape[:2], device=z.device, dtype=torch.bool),
        ], dim=1)
        width = max(z.shape[1], candidates.shape[1])
        old = F.pad(z, (0, 0, 0, width - z.shape[1]))
        new = old.clone()
        new[gate] = F.pad(candidates, (0, 0, 0, width - candidates.shape[1]))
        new_valid = F.pad(valid, (0, width - valid.shape[1]), value=False)
        new_valid[gate] = F.pad(candidate_valid, (0, width - candidate_valid.shape[1]), value=False)
        tcm_state['valid_mask'] = new_valid
        return new, tcm_state, True

    @staticmethod
    def _flatten_cls(cls_logit: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """
        支持 cls_logit 形状：
          - (B,1,H,W)
          - (B,H,W)
          - (B,N)
        返回：
          cls_flat: (B,N)
          meta: 用于 reshape 回去的信息
        """
        if cls_logit.dim() == 4:
            # (B,1,H,W) -> (B,N)
            B, _, H, W = cls_logit.shape
            cls_flat = cls_logit.flatten(2).squeeze(1)
            return cls_flat, ("B1HW", B, H, W)
        elif cls_logit.dim() == 3:
            # (B,H,W) -> (B,N)
            B, H, W = cls_logit.shape
            cls_flat = cls_logit.flatten(1)
            return cls_flat, ("BHW", B, H, W)
        elif cls_logit.dim() == 2:
            # (B,N)
            return cls_logit, ("BN",)
        else:
            raise ValueError(f"Unsupported cls_logit dim={cls_logit.dim()}, shape={tuple(cls_logit.shape)}")

    @staticmethod
    def _unflatten_cls(cls_flat: torch.Tensor, meta: tuple) -> torch.Tensor:
        """
        把 (B,N) reshape 回输入的 cls_logit 形状
        """
        tag = meta[0]
        if tag == "B1HW":
            _, B, H, W = meta
            return cls_flat.view(B, 1, H, W)
        if tag == "BHW":
            _, B, H, W = meta
            return cls_flat.view(B, H, W)
        if tag == "BN":
            return cls_flat
        raise ValueError(f"Unknown meta tag: {tag}")

    def _estimate_reliability(self, cls_ref: torch.Tensor) -> torch.Tensor:
        """
        估计当前响应是否可靠。长序列里 softmax max 很容易对干扰物过度自信，
        因此这里混合 peak、空间尖锐度和 top1/top2 gap。
        """
        prob = cls_ref.sigmoid()
        peak = prob.max(dim=1).values

        spatial_prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(spatial_prob * spatial_prob.clamp_min(1e-6).log()).sum(dim=1)
        entropy = entropy / math.log(max(prob.shape[1], 2))
        sharpness = 1.0 - entropy.clamp(0.0, 1.0)

        topk = prob.topk(k=min(2, prob.shape[1]), dim=1).values
        if topk.shape[1] == 1:
            gap = torch.ones_like(peak)
        else:
            gap = (topk[:, 0] - topk[:, 1]).clamp(0.0, 1.0)

        return (0.45 * peak + 0.35 * sharpness + 0.20 * gap).clamp(0.0, 1.0)

    def forward(
        self,
        S: torch.Tensor,
        cls_logit: torch.Tensor,
        q_prev: Optional[torch.Tensor],
    ):
        """
        Args:
            S: (B, N, C) 当前帧 search tokens
            cls_logit: (B,1,H,W) or (B,H,W) or (B,N) 当前帧分类 logit/score
            q_prev: (B,2,C) 上一帧 short/long prompt；兼容旧的 (B,C)

        Returns:
            q_new: (B,2,C) 更新后的 short/long prompt
            cls_refined: 与 cls_logit 同形状（校准后的分类图）
            bias_map: 与 cls_logit 同形状（tanh(sim)）
            reliability: (B,) 当前帧可靠性
        """
        assert S.dim() == 3, f"S must be (B,N,C), got {tuple(S.shape)}"
        B, N, C = S.shape
        assert C == self.dim, f"CTP dim mismatch: S.C={C} vs self.dim={self.dim}"

        cls_flat, meta = self._flatten_cls(cls_logit)  # (B,N)
        assert cls_flat.shape[0] == B, "Batch size mismatch between S and cls_logit"
        assert cls_flat.shape[1] == N, f"N mismatch: cls_flat.N={cls_flat.shape[1]} vs S.N={N}"

        q_short, q_long = self._split_state(q_prev, B, S.device, S.dtype)
        q_eff = self.get_prompt(torch.stack([q_short, q_long], dim=1))

        # --- A) bias from previous prompt ---
        s_proj = self.proj_s(S)                 # (B,N,C)
        q_proj = self.proj_q(q_eff).unsqueeze(1)  # (B,1,C)
        sim = (s_proj * q_proj).sum(-1) / math.sqrt(C)  # (B,N)
        bias = torch.tanh(sim)                  # (B,N)

        warm_ref = cls_flat + self.alpha * bias
        reliability = self._estimate_reliability(warm_ref)
        cls_ref = cls_flat + self.alpha * reliability.unsqueeze(1) * bias  # (B,N)

        # --- B) pooling to update prompt ---
        w = F.softmax(cls_ref / self.tau, dim=1)       # (B,N)
        f = torch.einsum("bn,bnc->bc", w, S)           # (B,C)

        q_hat = self.gru(f, q_eff)                     # (B,C)

        # --- C) reliability-gated soft update ---
        short_gate_bool = reliability > self.conf_thr
        long_gate_bool = reliability > self.long_thr
        short_gate = short_gate_bool.to(S.dtype).unsqueeze(-1)
        long_gate = long_gate_bool.to(S.dtype).unsqueeze(-1)
        beta_short = (self.short_beta * reliability).unsqueeze(-1) * short_gate
        beta_long = (self.long_beta * reliability).unsqueeze(-1) * long_gate

        q_short_new = F.normalize((1.0 - beta_short) * q_short + beta_short * q_hat, dim=-1)
        q_long_new = F.normalize((1.0 - beta_long) * q_long + beta_long * q_hat.detach(), dim=-1)
        # A closed gate preserves even an unnormalized learnable initial state.
        q_short_new = torch.where(short_gate_bool[:, None], q_short_new, q_short)
        q_long_new = torch.where(long_gate_bool[:, None], q_long_new, q_long)
        q_new = torch.stack([q_short_new, q_long_new], dim=1)

        # reshape back
        cls_refined = self._unflatten_cls(cls_ref, meta)
        bias_map = self._unflatten_cls(bias, meta)

        return q_new, cls_refined, bias_map, reliability
