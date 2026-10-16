from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from src.models.dataclasses import PerplexityConfig, PerplexityProfile
from src.perplexity.dinucleotide import encode_dinucleotides, has_ambiguous_base
from src.perplexity.entropy import perplexity_from_entropy, shannon_entropy


CHUNK_SIZE = 50_000


def _raw_perplexity(sequence: str, config: PerplexityConfig) -> np.ndarray:
    ambiguous = has_ambiguous_base(sequence, config.perplexity_window, config.step_size)
    dinucleotide_indices = encode_dinucleotides(sequence, config.perplexity_window, config.step_size)

    valid_rows = np.flatnonzero(~ambiguous)
    if valid_rows.size:
        flat_indices = valid_rows[:, None] * 16 + dinucleotide_indices[valid_rows]
        counts = np.bincount(
            flat_indices.ravel(),
            minlength=len(dinucleotide_indices) * 16,
        ).reshape(-1, 16)
    else:
        counts = np.zeros((len(dinucleotide_indices), 16), dtype=np.int64)

    probs = counts / max(config.perplexity_window - 1, 1)
    entropy = shannon_entropy(probs)
    raw = perplexity_from_entropy(entropy).astype(np.float32)
    raw[ambiguous] = np.nan
    return raw


def calculate_perplexity_profile(
    sequence: str, config: PerplexityConfig, chunk_size: int = CHUNK_SIZE
) -> PerplexityProfile:
    if len(sequence) < config.perplexity_window:
        empty = np.array([], dtype=np.float32)
        return PerplexityProfile(positions=np.array([], dtype=np.int64), raw_perplexity=empty, smoothed_perplexity=empty)

    window, step = config.perplexity_window, config.step_size
    if len(sequence) <= chunk_size:
        raw = _raw_perplexity(sequence, config)
    else:
        # Process long sequences in chunks of ~chunk_size nt; chunks overlap by
        # window - step so every window is evaluated exactly once, unchanged.
        n_windows = (len(sequence) - window) // step + 1
        per_chunk = max(1, (chunk_size - window) // step + 1)
        parts = []
        for first in range(0, n_windows, per_chunk):
            last = min(first + per_chunk, n_windows)
            sub = sequence[first * step:(last - 1) * step + window]
            parts.append(_raw_perplexity(sub, config))
        raw = np.concatenate(parts)

    smoothed = smooth_profile(raw, config.smoothing_window, config.smoothing_poly_order)
    positions = np.arange(len(raw), dtype=np.int64) * config.step_size
    return PerplexityProfile(positions=positions, raw_perplexity=raw, smoothed_perplexity=smoothed)


def smooth_profile(values: np.ndarray, window: int, poly_order: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values.copy()
    wl = min(window, n)
    if wl % 2 == 0:
        wl -= 1
    po = min(poly_order, wl - 1)
    if wl < 3 or po < 1:
        return values.astype(np.float32)

    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()

    work = values.astype(np.float64)
    if not np.all(finite):
        xp = np.flatnonzero(finite)
        fp = values[finite].astype(np.float64)
        work = np.interp(np.arange(n, dtype=np.float64), xp, fp)

    smoothed = savgol_filter(work, wl, po).astype(np.float32)
    smoothed[~finite] = np.nan
    return smoothed
