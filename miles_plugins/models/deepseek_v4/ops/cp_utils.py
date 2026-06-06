"""
Utility functions for DeepSeek V4 Context Parallelism support.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


@lru_cache(1)
def _get_window_topk_idxs_ref(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Reference (single-device, no-CP) window topk index builder. Used only as
    an equality oracle by :func:`get_window_topk_idxs_cp` when ``cp_size == 1``.
    """

    def _inner():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1).cuda()


@lru_cache(2)
def _get_compress_topk_idxs_ref(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """Reference (single-device, no-CP) compress topk index builder. Used only as
    an equality oracle by :func:`get_compress_topk_idxs_cp` when ``cp_size == 1``.
    """

    def _inner():
        if start_pos > 0:
            return torch.arange(0, (start_pos + 1) // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1).cuda()


def all_gather_cp(tensor: Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    """All-gather tensor across CP ranks on `dim`. Contiguous CP = result already in natural order."""
    return torch.cat(torch.distributed.nn.functional.all_gather(tensor, group=cp_group), dim=dim)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> Tensor:
    """Get global positions for local q tokens (contiguous CP)."""
    if cp_size <= 1 or cp_group is None:
        return torch.arange(0, seqlen_local, device=device)
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return torch.arange(start, start + seqlen_local, device=device)


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get window topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    base = q_positions.unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen_global, window_size), device=device)
    topk_idxs = torch.where(k_pos > base, -1, k_pos)
    result = topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = _get_window_topk_idxs_ref(window_size, bsz, seqlen_local, start_pos=0)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_window_topk_idxs_cp mismatch with ref"

    return result


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get static compress topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    offset = seqlen_global
    k_group_idx = torch.arange(seqlen_global // ratio, device=device).repeat(seqlen_local, 1)
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    invalid_mask = k_group_idx >= q_first_invalid_group
    compress_topk_idxs = torch.where(invalid_mask, -1, k_group_idx + offset)
    result = compress_topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = _get_compress_topk_idxs_ref(ratio, bsz, seqlen_local, start_pos=0, offset=offset)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_compress_topk_idxs_cp mismatch with ref"

    return result


def get_freqs_cis_for_cp(
    freqs_cis: Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> Tensor:
    """Get freqs_cis for this CP rank (contiguous slice)."""
    if cp_size == 1 or cp_group is None:
        return freqs_cis[:seqlen_local:stride]
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return freqs_cis[start : start + seqlen_local : stride]


def get_packed_zigzag_positions_for_cp(
    cu_seqlens: Tensor,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup | None,
    local_seqlen: int,
) -> Tensor:
    """Build per-token RoPE positions for Miles packed THD + zigzag CP.

    Miles' default THD CP slicer takes two chunks from each sample on every CP
    rank: the rank's forward chunk and its mirrored chunk from the end of the
    padded sequence. The V4 attention code cannot use a single contiguous
    ``cp_rank * local_seqlen`` offset for packed micro-batches because positions
    must reset at every packed sample boundary.

    ``cu_seqlens`` uses original/global sequence lengths in PackedSeqParams. In
    the non-allgather CP path those lengths are exactly ``local_len * cp_size``
    for each packed segment.
    """

    cu_seqlens = cu_seqlens.to(dtype=torch.long)
    device = cu_seqlens.device

    if cp_size <= 1 or cp_group is None:
        local_cu_seqlens = cu_seqlens
        idx = torch.arange(local_seqlen, device=device, dtype=torch.long)
        segment = torch.searchsorted(local_cu_seqlens[1:], idx, right=True)
        return idx - local_cu_seqlens[segment]

    if bool((cu_seqlens % cp_size != 0).any().item()):
        raise RuntimeError(
            f"Packed THD cu_seqlens must be divisible by cp_size={cp_size}; got {cu_seqlens.tolist()}"
        )

    local_cu_seqlens = cu_seqlens // cp_size
    if int(local_cu_seqlens[-1].item()) != local_seqlen:
        raise RuntimeError(
            "Packed THD local sequence length mismatch: "
            f"cu_seqlens[-1] / cp_size = {int(local_cu_seqlens[-1].item())}, "
            f"but hidden_states local_seqlen = {local_seqlen}"
        )

    local_lengths = local_cu_seqlens[1:] - local_cu_seqlens[:-1]
    if bool((local_lengths % 2 != 0).any().item()):
        raise RuntimeError(
            "Miles zigzag CP expects each packed local segment to contain two equal chunks; "
            f"got local lengths {local_lengths.tolist()}"
        )

    cp_rank = cp_group.rank()
    idx = torch.arange(local_seqlen, device=device, dtype=torch.long)
    segment = torch.searchsorted(local_cu_seqlens[1:], idx, right=True)
    segment_start = local_cu_seqlens[segment]
    local_offset = idx - segment_start
    chunk = local_lengths[segment] // 2

    return torch.where(
        local_offset < chunk,
        cp_rank * chunk + local_offset,
        (2 * cp_size - cp_rank - 1) * chunk + (local_offset - chunk),
    )


def get_freqs_cis_for_positions(freqs_cis: Tensor, positions: Tensor, *, stride: int = 1) -> Tensor:
    """Index RoPE frequencies by explicit per-token positions."""

    if stride != 1:
        positions = positions[::stride]

    if positions.numel() == 0:
        return freqs_cis[:0]

    max_position = int(positions.max().item())
    if max_position >= freqs_cis.size(0):
        raise RuntimeError(
            f"RoPE position {max_position} exceeds precomputed freqs_cis length {freqs_cis.size(0)}"
        )

    return freqs_cis.index_select(0, positions.to(device=freqs_cis.device, dtype=torch.long))
