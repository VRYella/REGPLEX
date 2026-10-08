import numpy as np

from src.models.dataclasses import PerplexityConfig
from src.perplexity.profile import calculate_perplexity_profile


def test_short_sequence_returns_empty():
    cfg = PerplexityConfig(perplexity_window=10)
    profile = calculate_perplexity_profile("ACGT", cfg)
    assert profile.raw_perplexity.size == 0


def test_ambiguous_base_sets_nan():
    cfg = PerplexityConfig(perplexity_window=5)
    profile = calculate_perplexity_profile("AAAAANAAAAA", cfg)
    assert np.isnan(profile.raw_perplexity).any()


def test_single_dinucleotide_low_perplexity():
    cfg = PerplexityConfig(perplexity_window=6)
    profile = calculate_perplexity_profile("AAAAAAAAAAAA", cfg)
    finite = profile.raw_perplexity[np.isfinite(profile.raw_perplexity)]
    assert finite.size > 0
    assert np.allclose(finite, 1.0)


def test_large_window_counts_do_not_overflow():
    cfg = PerplexityConfig(perplexity_window=40_000)
    profile = calculate_perplexity_profile("A" * 40_001, cfg)
    assert np.allclose(profile.raw_perplexity, 1.0)
