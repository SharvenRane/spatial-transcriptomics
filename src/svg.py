"""Detect spatially variable genes with Moran's I.

A spatially variable gene is one whose expression is autocorrelated in space:
nearby spots tend to have similar values. Moran's I measures exactly that. We
build a spatial weight matrix from the spot coordinates (k nearest neighbours),
then score each gene by its Moran's I. Genes with planted spatial structure
should score well above genes whose expression ignores position.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def spatial_weights(coords: np.ndarray, n_neighbors: int = 6) -> np.ndarray:
    """Build a row normalized k nearest neighbour spatial weight matrix.

    Parameters
    ----------
    coords : (n_spots, 2) array
        Spot coordinates.
    n_neighbors : int
        Number of spatial neighbours per spot (excluding itself).

    Returns
    -------
    W : (n_spots, n_spots) array
        Weight matrix with zero diagonal, each row summing to one (rows with no
        neighbours, which cannot happen for k >= 1, would sum to zero).
    """
    coords = np.asarray(coords, dtype=float)
    n_spots = coords.shape[0]
    k = min(n_neighbors, n_spots - 1)
    if k < 1:
        raise ValueError("need at least two spots to build spatial weights")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)

    W = np.zeros((n_spots, n_spots), dtype=float)
    for i in range(n_spots):
        # idx[i, 0] is the spot itself; skip it.
        neighbours = idx[i, 1:]
        W[i, neighbours] = 1.0

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    return W


def moran_i(x: np.ndarray, W: np.ndarray) -> float:
    """Compute Moran's I spatial autocorrelation for one gene.

    Parameters
    ----------
    x : (n_spots,) array
        Expression values for one gene across spots.
    W : (n_spots, n_spots) array
        Spatial weight matrix.

    Returns
    -------
    float
        Moran's I. Near +1 means strong positive spatial autocorrelation,
        near 0 means none, negative means a checkerboard like anti pattern.
        Returns 0.0 for a constant gene (no variance to be structured).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    z = x - x.mean()
    denom = np.sum(z * z)
    if denom == 0:
        return 0.0
    s0 = W.sum()
    if s0 == 0:
        return 0.0
    num = z @ (W @ z)
    return float((n / s0) * (num / denom))


def rank_spatially_variable_genes(
    counts: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 6,
    log_transform: bool = True,
):
    """Score every gene by Moran's I and return them ranked.

    Parameters
    ----------
    counts : (n_spots, n_genes) array
        Expression counts.
    coords : (n_spots, 2) array
        Spot coordinates.
    n_neighbors : int
        Neighbours used to build the spatial weights.
    log_transform : bool
        Apply log1p before scoring, which stabilizes Poisson count variance.

    Returns
    -------
    scores : (n_genes,) array
        Moran's I per gene, in original gene order.
    order : (n_genes,) int array
        Gene indices sorted by descending Moran's I.
    """
    counts = np.asarray(counts, dtype=float)
    if log_transform:
        counts = np.log1p(counts)

    W = spatial_weights(coords, n_neighbors=n_neighbors)
    n_genes = counts.shape[1]
    scores = np.empty(n_genes, dtype=float)
    for g in range(n_genes):
        scores[g] = moran_i(counts[:, g], W)

    order = np.argsort(-scores)
    return scores, order
