"""BERT encoder attention adapter for transformers==5.3.0.

Uses the inference-only Ascend FIA API. CPU tests inject a mathematical
reference; production never silently falls back to that reference.
"""
import math
import torch
from torch import nn


def npu_fia(q, k, v, blocked, heads, scale):
    import torch_npu
    return torch_npu.npu_fused_infer_attention_score(
        q.contiguous(), k.contiguous(), v.contiguous(),
        num_heads=heads, num_key_value_heads=heads,
        input_layout="BNSD", scale=scale,
        atten_mask=blocked,
        sparse_mode=0, pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False,
    )[0]


class FusedBertSelfAttention(nn.Module):
    """Preserves Q/K/V weights and BERT's (context, attention_weights) contract."""
    def __init__(self, original, kernel=npu_fia):
        super().__init__()
        if original.is_decoder or getattr(original, "is_causal", False):
            raise ValueError("Only bidirectional encoder self-attention is supported")
        self.query = original.query
        self.key = original.key
        self.value = original.value
        self.heads = original.num_attention_heads
        self.dim = original.attention_head_size
        self.scale = 1.0 / math.sqrt(self.dim)
        self.kernel = kernel

    def forward(self, hidden_states, attention_mask=None,
                past_key_values=None, cache_position=None,
                encoder_hidden_states=None, **kwargs):
        if self.training:
            raise ValueError("This adapter is inference-only; call eval()")
        if past_key_values is not None or encoder_hidden_states is not None:
            raise ValueError("KV cache and cross-attention are not supported")
        if kwargs.get("head_mask") is not None or kwargs.get("output_attentions", False):
            raise ValueError("head_mask / attention probability output is unsupported")
        b, s, _ = hidden_states.shape
        q = self.query(hidden_states).reshape(b, s, self.heads, self.dim).transpose(1, 2)
        k = self.key(hidden_states).reshape(b, s, self.heads, self.dim).transpose(1, 2)
        v = self.value(hidden_states).reshape(b, s, self.heads, self.dim).transpose(1, 2)
        blocked = None
        if attention_mask is not None:
            # HF eager BERT supplies an additive floating mask: 0=keep, negative=block.
            # The FIA bool mask has the opposite meaning to the tokenizer's 1=keep.
            if not attention_mask.is_floating_point() or attention_mask.ndim != 4:
                raise ValueError("Expected HF eager additive 4-D attention mask")
            blocked = (attention_mask < 0).expand(b, 1, s, s).contiguous()
        result = self.kernel(q, k, v, blocked, self.heads, self.scale)
        context = result.transpose(1, 2).contiguous().reshape(b, s, self.heads * self.dim)
        return context, None


def replace_bert_attention(model, kernel=npu_fia):
    if model.config.model_type != "bert":
        raise ValueError("This adapter supports BertModel only")
    if model.config.is_decoder or getattr(model.config, "add_cross_attention", False):
        raise ValueError("Decoder/cross-attention BERT is outside this lab")
    if getattr(model.config, "position_embedding_type", "absolute") != "absolute":
        raise ValueError("Relative position attention is unsupported")
    count = 0
    for layer in model.encoder.layer:
        layer.attention.self = FusedBertSelfAttention(layer.attention.self, kernel)
        count += 1
    model.eval()
    return count
