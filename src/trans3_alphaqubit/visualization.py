# visualization.py ok
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Dict, Any


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def plot_attention_heads(
    attn_weights: torch.Tensor,
    *,
    layer_idx: int = 0,
    batch_idx: int = 0,
    max_heads: Optional[int] = None,
    cmap: str = "hot",
):
    """
    Plot full attention matrices for each head.

    attn_weights: (B, H, L, L)
    """
    assert attn_weights.dim() == 4, f"expected (B,H,L,L), got {tuple(attn_weights.shape)}"
    B, H, L, _ = attn_weights.shape
    assert 0 <= batch_idx < B, f"batch_idx={batch_idx} out of range B={B}"

    data = _to_numpy(attn_weights[batch_idx])  # (H,L,L)

    if max_heads is not None:
        H_plot = min(H, int(max_heads))
        data = data[:H_plot]
    else:
        H_plot = H

    fig, axes = plt.subplots(1, H_plot, figsize=(5 * H_plot, 5), squeeze=False)
    fig.suptitle(f"Layer {layer_idx} Attention Maps (B={batch_idx})", fontsize=14)

    for h in range(H_plot):
        ax = axes[0, h]
        im = ax.imshow(data[h], cmap=cmap, interpolation="nearest")
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key (attend to)")
        ax.set_ylabel("Query (current token)")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    plt.tight_layout()
    plt.show()


def plot_token_attention_vector(
    attn_weights: torch.Tensor,
    *,
    token_idx: int,
    head_idx: int = 0,
    batch_idx: int = 0,
    layer_idx: int = 0,
    cmap: str = "viridis",
):
    """
    Plot a single query token's attention over keys as a 1D heatmap.

    attn_weights: (B,H,L,L)
    Shows attn_weights[batch, head, token_idx, :] -> (L,)
    """
    assert attn_weights.dim() == 4, f"expected (B,H,L,L), got {tuple(attn_weights.shape)}"
    B, H, L, _ = attn_weights.shape
    assert 0 <= batch_idx < B
    assert 0 <= head_idx < H
    assert 0 <= token_idx < L

    vec = _to_numpy(attn_weights[batch_idx, head_idx, token_idx, :])  # (L,)

    plt.figure(figsize=(10, 1.6))
    plt.imshow(vec[None, :], aspect="auto", cmap=cmap)
    plt.title(f"Layer {layer_idx} Head {head_idx}: query token {token_idx} attends to keys")
    plt.yticks([])
    plt.xlabel("Key index")
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_token_attention_on_grid(
    attn_weights: torch.Tensor,
    *,
    token_idx: int,
    head_idx: int = 0,
    batch_idx: int = 0,
    layer_idx: int = 0,
    grid_hw: Optional[Tuple[int, int]] = None,
    index_to_coord: Optional[Sequence[Tuple[int, int]]] = None,
    fill_value: float = 0.0,
    cmap: str = "viridis",
):
    """
    Plot "what token_idx sees" on a 2D grid.

    You have TWO safe options:

    A) If you truly know L == H*W (dense grid tokens):
        pass grid_hw=(H,W) and index_to_coord=None

    B) If tokens correspond to sparse stabilizers on a (d+1)x(d+1) grid:
        pass index_to_coord (length L) mapping token index -> (i,j),
        and pass grid_hw=(H,W). Unmapped cells use fill_value.

    attn_weights: (B,H,L,L)
    """
    assert attn_weights.dim() == 4, f"expected (B,H,L,L), got {tuple(attn_weights.shape)}"
    B, Hh, L, _ = attn_weights.shape
    assert 0 <= batch_idx < B
    assert 0 <= head_idx < Hh
    assert 0 <= token_idx < L

    if grid_hw is None:
        raise ValueError("grid_hw must be provided, e.g. grid_hw=(d+1,d+1).")

    H_grid, W_grid = grid_hw
    vec = _to_numpy(attn_weights[batch_idx, head_idx, token_idx, :])  # (L,)

    if index_to_coord is None:
        # Dense grid case: require L == H*W
        if L != H_grid * W_grid:
            raise ValueError(f"Dense grid requires L == H*W, but L={L}, H*W={H_grid*W_grid}. "
                             f"Provide index_to_coord for sparse mapping.")
        grid = vec.reshape(H_grid, W_grid)
    else:
        if len(index_to_coord) != L:
            raise ValueError(f"index_to_coord length must equal L. got len={len(index_to_coord)} L={L}")
        grid = np.full((H_grid, W_grid), fill_value, dtype=np.float32)
        for k, (i, j) in enumerate(index_to_coord):
            if 0 <= i < H_grid and 0 <= j < W_grid:
                grid[i, j] = vec[k]

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    plt.title(f"Layer {layer_idx} Head {head_idx}: token {token_idx} attention on grid")
    plt.colorbar(shrink=0.8)
    plt.xlabel("grid j")
    plt.ylabel("grid i")
    plt.tight_layout()
    plt.show()
