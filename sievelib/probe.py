"""
probe.py -- capture post-RoPE / post-QK-norm QUERIES during decode.

We deliberately capture ONLY the query here. Keys and values are read afterwards
from the model's own KV cache, which keeps the forward pass cheap and avoids
holding [H, L, d] tensors inside the hook.

Registering into ALL_ATTENTION_FUNCTIONS (transformers >= 4.48) means the tensors
handed to us are already post-RoPE and post-QK-norm in the model's own convention,
so the probe is architecture-agnostic: Llama, Qwen3 (QK-norm), Gemma (softcapping),
and MoE models all work unmodified.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError as e:  # pragma: no cover
    raise ImportError("needs transformers>=4.48; pip install -U transformers") from e


class ProbeState:
    def __init__(self):
        self.enabled = False
        self.q = {}          # layer_idx -> [H, d] float32 cpu (post-RoPE query)
        self.scaling = {}    # layer_idx -> float
        self.mask = {}       # layer_idx -> [L] additive mask or None
        self.ref_out = {}    # layer_idx -> [H, d] the model's own attention output

        self._counter = 0

    def reset(self):
        self.q.clear(); self.scaling.clear(); self.mask.clear(); self.ref_out.clear()
        self._counter = 0

    def next_key(self, layer_idx):
        """layer_idx can be None on some architectures. Falling back to a single
        None key would silently collapse every layer into one entry, so we assign
        a monotonic index instead."""
        if layer_idx is None:
            layer_idx = self._counter
        self._counter += 1
        return layer_idx


STATE = ProbeState()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _sdpa(query, key, value, attention_mask, scaling, is_causal):
    n_rep = query.shape[1] // key.shape[1]
    k, v = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    q_len, k_len = query.shape[2], k.shape[-2]
    mask = attention_mask[:, :, :, :k_len] if attention_mask is not None else None

    # [REGRESSION] F.scaled_dot_product_attention(is_causal=True) applies a
    # TOP-LEFT aligned causal mask: query i sees keys 0..i. That is only correct
    # when q_len == k_len. During chunk 2+ of a chunked prefill the cache is
    # already populated, so a query at absolute position past_len + i must see
    # keys 0..past_len+i -- top-left alignment instead showed it keys 0..i, i.e.
    # almost nothing, misaligned. Every H0 run before this fix prefilled >4096
    # tokens in 4096-token chunks and so built its KV cache through this path.
    #
    # It hid from both validation levels at once: L1/L3 exercise the probe but in
    # a SINGLE forward pass (past_len == 0, where top-left is correct), and
    # test_chunked_prefill chunks but builds the model with attn_implementation
    # "eager", never reaching this function.
    if mask is None and is_causal and q_len != k_len:
        pos = torch.arange(q_len, device=query.device).unsqueeze(-1) + (k_len - q_len)
        j = torch.arange(k_len, device=query.device).unsqueeze(0)
        mask = torch.zeros(q_len, k_len, dtype=query.dtype, device=query.device)
        mask.masked_fill_(j > pos, torch.finfo(query.dtype).min)
        mask = mask[None, None]

    out = F.scaled_dot_product_attention(
        query, k, v, attn_mask=mask, dropout_p=0.0, scale=scaling,
        is_causal=(is_causal and mask is None and q_len == k_len))
    return out.transpose(1, 2).contiguous()


def sieve_probe_attention(module, query, key, value, attention_mask=None,
                          scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    q_len = query.shape[2]
    out = _sdpa(query, key, value, attention_mask, scaling, q_len > 1)

    if STATE.enabled and q_len == 1:
        li = STATE.next_key(getattr(module, "layer_idx", None))
        STATE.q[li] = query[0, :, 0].detach().float().cpu()
        STATE.scaling[li] = float(scaling)
        STATE.mask[li] = (attention_mask[0, 0, 0].detach().float().cpu()
                          if attention_mask is not None else None)
        STATE.ref_out[li] = out[0, 0].detach().float().cpu()   # [H, d]
    return out, None


def install():
    ALL_ATTENTION_FUNCTIONS["sieve_probe"] = sieve_probe_attention


# --------------------------------------------------------------------- KV cache
def cache_kv(past, layer_idx):
    """Return (K, V) as [Hkv, L, d] for one layer, across transformers versions."""
    if hasattr(past, "layers"):                      # transformers >= 4.54
        lay = past.layers[layer_idx]
        k, v = lay.keys, lay.values
    elif hasattr(past, "key_cache"):                 # DynamicCache (older)
        k, v = past.key_cache[layer_idx], past.value_cache[layer_idx]
    else:                                            # legacy tuple
        k, v = past[layer_idx][0], past[layer_idx][1]
    return k[0], v[0]


def n_layers_in_cache(past):
    if hasattr(past, "layers"):
        return len(past.layers)
    if hasattr(past, "key_cache"):
        return len(past.key_cache)
    return len(past)