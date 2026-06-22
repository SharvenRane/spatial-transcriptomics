"""Cluster spatial transcriptomics spots by their expression profiles.

The pipeline mirrors common single cell practice: normalize each spot for
sequencing depth, log transform, scale features, reduce with PCA, then cluster
in the reduced space with KMeans. A small Hungarian style matcher maps the
arbitrary cluster ids back onto the ground truth domain ids so accuracy can be
measured.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Library size normalize then log1p transform raw counts.

    Each spot is scaled so its total expression equals ``target_sum`` before
    the log transform. This removes per spot depth differences that would
    otherwise dominate the clustering.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    normed = counts / totals * target_sum
    return np.log1p(normed)


def cluster_spots(
    counts: np.ndarray,
    n_clusters: int,
    n_components: int = 10,
    seed: int = 0,
):
    """Cluster spots into ``n_clusters`` groups from their expression.

    Parameters
    ----------
    counts : (n_spots, n_genes) array
        Raw expression counts.
    n_clusters : int
        Number of clusters to form (usually the number of domains).
    n_components : int
        PCA dimensionality. Capped at the number of genes and spots.
    seed : int
        Random seed for PCA and KMeans.

    Returns
    -------
    labels : (n_spots,) int array
        Cluster assignment per spot.
    """
    counts = np.asarray(counts, dtype=float)
    n_spots, n_genes = counts.shape
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")

    normed = normalize_counts(counts)
    scaled = StandardScaler().fit_transform(normed)

    max_comp = min(n_components, n_genes, n_spots)
    embedding = PCA(n_components=max_comp, random_state=seed).fit_transform(scaled)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)
    return labels


def match_clusters_to_domains(
    cluster_labels: np.ndarray,
    domain_labels: np.ndarray,
):
    """Map cluster ids onto ground truth domain ids by best overlap.

    Builds a contingency table between predicted clusters and true domains and
    solves the assignment that maximizes the count on the diagonal. Returns the
    remapped labels (so a remapped label equals the matched domain id) along
    with the accuracy of that matching.

    Returns
    -------
    remapped : (n_spots,) int array
        Cluster labels relabeled to align with domain ids.
    accuracy : float
        Fraction of spots whose remapped label equals the true domain.
    mapping : dict
        cluster id -> domain id.
    """
    cluster_labels = np.asarray(cluster_labels)
    domain_labels = np.asarray(domain_labels)

    clusters = np.unique(cluster_labels)
    domains = np.unique(domain_labels)
    n = max(clusters.max(), domains.max()) + 1

    contingency = np.zeros((n, n), dtype=int)
    for c, d in zip(cluster_labels, domain_labels):
        contingency[c, d] += 1

    # Maximize overlap -> minimize negative overlap.
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    remapped = np.array([mapping[int(c)] for c in cluster_labels])
    accuracy = float(np.mean(remapped == domain_labels))
    return remapped, accuracy, mapping
