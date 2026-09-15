"""Assembling a multimodal prompt.

Two things have to line up for an image to work: the token sequence must
reserve exactly one placeholder per merged patch, and the position ids must
give those placeholders their true (row, column) so mRoPE can tell the model
where in the picture each token came from.
"""
from __future__ import annotations

import torch

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


def build_prompt(tokenizer, n_image_tokens: int, question: str,
                 enable_thinking: bool | None = None) -> str:
    """ChatML with a run of image placeholders the tower will fill in."""
    body = (
        VISION_START + IMAGE_PAD * n_image_tokens + VISION_END + question
    )
    extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True, **extra
    )


def position_ids(
    ids: list[int],
    image_token_id: int,
    grid_h: int,
    grid_w: int,
    device,
) -> torch.Tensor:
    """(3, T) temporal/height/width positions.

    Text advances all three axes together. Image tokens share one temporal
    position and spread across height and width, then the running counter
    jumps past the larger of the two so nothing collides afterwards.
    """
    t_ids, h_ids, w_ids = [], [], []
    cur = 0
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] != image_token_id:
            t_ids.append(cur)
            h_ids.append(cur)
            w_ids.append(cur)
            cur += 1
            i += 1
            continue
        span = 0
        while i + span < n and ids[i + span] == image_token_id:
            span += 1
        for k in range(span):
            t_ids.append(cur)
            h_ids.append(cur + (k // grid_w))
            w_ids.append(cur + (k % grid_w))
        cur += max(grid_h, grid_w)
        i += span
    return torch.tensor([t_ids, h_ids, w_ids], dtype=torch.long, device=device)


def next_position(pos: torch.Tensor) -> torch.Tensor:
    """Position ids for one appended decode token."""
    nxt = int(pos.max()) + 1
    return torch.tensor([[nxt], [nxt], [nxt]], dtype=torch.long,
                        device=pos.device)
