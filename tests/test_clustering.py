import numpy as np

from src.synthetic import make_synthetic_tissue
from src.clustering import (
    cluster_spots,
    match_clusters_to_domains,
    normalize_counts,
)


def test_normalize_counts_equalizes_depth():
    counts = np.array([[1.0, 1.0, 2.0], [10.0, 10.0, 20.0]])
    normed = normalize_counts(counts)
    # After depth normalization the two spots have identical relative profiles,
    # so their log normalized rows should match.
    assert np.allclose(normed[0], normed[1])


def test_clustering_recovers_known_domains():
    t = make_synthetic_tissue(grid_size=20, n_patterned=12, n_random=36, seed=0)
    labels = cluster_spots(t.counts, n_clusters=t.n_domains, seed=0)
    remapped, accuracy, mapping = match_clusters_to_domains(
        labels, t.domain_labels
    )
    # The planted four domain structure should be recovered well above chance
    # (chance with four balanced domains is 0.25).
    assert accuracy > 0.85


def test_cluster_count_matches_request():
    t = make_synthetic_tissue(grid_size=16, seed=4)
    labels = cluster_spots(t.counts, n_clusters=t.n_domains, seed=4)
    assert len(np.unique(labels)) == t.n_domains


def test_mapping_is_a_permutation():
    t = make_synthetic_tissue(grid_size=18, seed=2)
    labels = cluster_spots(t.counts, n_clusters=t.n_domains, seed=2)
    _, _, mapping = match_clusters_to_domains(labels, t.domain_labels)
    # Each cluster maps to a distinct domain id.
    assert len(set(mapping.values())) == len(mapping)


def test_clustering_beats_chance_across_seeds():
    accs = []
    for s in range(3):
        t = make_synthetic_tissue(grid_size=20, seed=s)
        labels = cluster_spots(t.counts, n_clusters=t.n_domains, seed=s)
        _, acc, _ = match_clusters_to_domains(labels, t.domain_labels)
        accs.append(acc)
    assert min(accs) > 0.7
