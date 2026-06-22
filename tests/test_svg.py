import numpy as np

from src.synthetic import make_synthetic_tissue
from src.svg import (
    moran_i,
    spatial_weights,
    rank_spatially_variable_genes,
)


def test_spatial_weights_row_normalized():
    t = make_synthetic_tissue(grid_size=10, seed=0)
    W = spatial_weights(t.coords, n_neighbors=6)
    n = t.n_spots
    assert W.shape == (n, n)
    # Zero diagonal, rows sum to one.
    assert np.allclose(np.diag(W), 0.0)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_moran_constant_gene_is_zero():
    t = make_synthetic_tissue(grid_size=8, seed=0)
    W = spatial_weights(t.coords)
    x = np.full(t.n_spots, 3.0)
    assert moran_i(x, W) == 0.0


def test_moran_high_for_smooth_gradient():
    # A smooth spatial gradient is strongly autocorrelated; Moran's I near 1.
    t = make_synthetic_tissue(grid_size=20, seed=0)
    W = spatial_weights(t.coords)
    gradient = t.coords[:, 0] + t.coords[:, 1]
    assert moran_i(gradient, W) > 0.8


def test_moran_low_for_white_noise():
    t = make_synthetic_tissue(grid_size=20, seed=0)
    W = spatial_weights(t.coords)
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(t.n_spots)
    # White noise has no spatial structure, Moran's I near zero.
    assert abs(moran_i(noise, W)) < 0.15


def test_patterned_genes_rank_above_random():
    t = make_synthetic_tissue(grid_size=20, n_patterned=12, n_random=36, seed=0)
    scores, order = rank_spatially_variable_genes(t.counts, t.coords)

    pat_scores = scores[t.gene_is_patterned]
    rnd_scores = scores[~t.gene_is_patterned]
    # Planted genes should score clearly higher on average.
    assert pat_scores.mean() > rnd_scores.mean()

    # The top patterned genes should dominate the top of the ranking. With 12
    # patterned genes, the top 12 ranked genes should be mostly patterned.
    n_pat = int(t.gene_is_patterned.sum())
    top = order[:n_pat]
    hits = t.gene_is_patterned[top].sum()
    assert hits >= n_pat - 1  # allow at most one intruder


def test_every_patterned_gene_beats_random_max():
    # Stronger property: with strong planted signal, the weakest patterned gene
    # still outscores the strongest random gene.
    t = make_synthetic_tissue(
        grid_size=20,
        n_patterned=12,
        n_random=36,
        pattern_strength=8.0,
        seed=1,
    )
    scores, _ = rank_spatially_variable_genes(t.counts, t.coords)
    pat_min = scores[t.gene_is_patterned].min()
    rnd_max = scores[~t.gene_is_patterned].max()
    assert pat_min > rnd_max
