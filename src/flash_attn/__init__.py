"""Small compatibility surface for environments without flash-attn.

TraceSearch uses veRL's padding helpers with SDPA.  The helper functions live
under ``flash_attn.bert_padding`` in veRL's CUDA path, even when no
FlashAttention kernel is selected.  The actual kernels are deliberately not
provided here.
"""

