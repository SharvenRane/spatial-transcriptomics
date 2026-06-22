import numpy as np

from src.synthetic import make_synthetic_tissue


def test_shapes_and_types():
    t = make_synthetic_tissue(grid_size=16, n_patterned=8, n_random=20, seed=1)
    assert t.counts.shape == (16 * 16, 8 + 20)
    assert t.coords.shape == (16 * 16, 2)
    assert t.domain_labels.shape == (16 * 16,)
    assert t.gene_is_patterned.shape == (28,)
    assert t.gene_is_patterned.sum() == 8
    assert len(t.gene_names) == 28


def test_counts_are_nonnegative_integers():
    t = make_synthetic_tissue(seed=2)
    assert np.all(t.counts >= 0)
    # Poisson draws stored as float should be whole numbers.
    assert np.allclose(t.counts, np.round(t.counts))


def test_four_contiguous_domains():
    t = make_synthetic_tissue(grid_size=20, seed=3)
    assert t.n_domains == 4
    # Every domain should occupy a quarter of a 20x20 grid.
    counts = np.bincount(t.domain_labels)
    assert np.all(counts == 100)


def test_reproducible_with_seed():
    a = make_synthetic_tissue(seed=7)
    b = make_synthetic_tissue(seed=7)
    assert np.array_equal(a.counts, b.counts)


def test_patterned_genes_are_more_structured_in_mean():
    # Patterned genes should show a larger spread of per domain means than
    # random genes, because their rate jumps in one domain.
    t = make_synthetic_tissue(grid_size=20, n_patterned=12, n_random=36, seed=5)
    spreads = []
    for g in range(t.n_genes):
        domain_means = [
            t.counts[t.domain_labels == d, g].mean() for d in range(t.n_domains)
        ]
        spreads.append(np.std(domain_means))
    spreads = np.array(spreads)
    pat = spreads[t.gene_is_patterned].mean()
    rnd = spreads[~t.gene_is_patterned].mean()
    assert pat > rnd
