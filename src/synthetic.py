"""Generate a synthetic spatial transcriptomics dataset.

The tissue is a regular square grid of spots. Each spot belongs to one of a
small number of spatial domains laid out as contiguous regions. Two kinds of
genes are planted:

  Patterned genes  : expression depends on the spatial domain a spot sits in,
                     so the gene is high in some domains and low in others.
                     These are the genes a spatially variable gene test should
                     surface.

  Random genes     : expression is drawn independently of position, with the
                     same marginal mean and variance scale as the patterned
                     genes so the test cannot cheat on magnitude alone.

Counts are produced from a Poisson model so the data looks like real UMI
counts (non negative integers) rather than smooth gaussians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticTissue:
    """Container for one synthetic tissue sample.

    Attributes
    ----------
    counts : (n_spots, n_genes) float array
        Expression counts per spot per gene.
    coords : (n_spots, 2) float array
        Spatial (x, y) coordinates of each spot.
    domain_labels : (n_spots,) int array
        Ground truth spatial domain for each spot.
    gene_is_patterned : (n_genes,) bool array
        True for genes whose expression was planted to follow the domains.
    gene_names : list of str
        Human readable names, patterned genes prefixed "pat", random "rnd".
    grid_shape : tuple of int
        (rows, cols) of the spot grid.
    """

    counts: np.ndarray
    coords: np.ndarray
    domain_labels: np.ndarray
    gene_is_patterned: np.ndarray
    gene_names: list
    grid_shape: tuple

    @property
    def n_spots(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    @property
    def n_domains(self) -> int:
        return int(self.domain_labels.max()) + 1


def _quadrant_domains(rows: int, cols: int) -> np.ndarray:
    """Assign each grid cell to one of four contiguous quadrant domains.

    Returns a (rows * cols,) int array in row major order.
    """
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    top = yy < rows / 2.0
    left = xx < cols / 2.0
    # domain ids: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
    domain = np.where(
        top & left, 0,
        np.where(top & ~left, 1, np.where(~top & left, 2, 3)),
    )
    return domain.reshape(-1)


def make_synthetic_tissue(
    grid_size: int = 20,
    n_patterned: int = 12,
    n_random: int = 36,
    base_rate: float = 4.0,
    pattern_strength: float = 6.0,
    seed: int = 0,
) -> SyntheticTissue:
    """Build a synthetic tissue grid with planted spatial structure.

    Parameters
    ----------
    grid_size : int
        Side length of the square spot grid. Total spots is grid_size ** 2.
    n_patterned : int
        Number of genes whose mean expression tracks the spatial domain.
    n_random : int
        Number of genes with position independent expression.
    base_rate : float
        Baseline Poisson rate (low expression level) shared by all genes.
    pattern_strength : float
        Extra Poisson rate added in the domains where a patterned gene is on.
        Larger values make the planted signal easier to recover.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    SyntheticTissue
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if n_patterned < 1:
        raise ValueError("need at least one patterned gene")
    if n_random < 0:
        raise ValueError("n_random must be non negative")

    rng = np.random.default_rng(seed)
    rows = cols = grid_size
    n_spots = rows * cols

    # Spatial coordinates on an integer grid, row major to match domains.
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    coords = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(float)

    domain_labels = _quadrant_domains(rows, cols)
    n_domains = int(domain_labels.max()) + 1

    n_genes = n_patterned + n_random
    rates = np.full((n_spots, n_genes), base_rate, dtype=float)

    gene_is_patterned = np.zeros(n_genes, dtype=bool)
    gene_is_patterned[:n_patterned] = True

    # Each patterned gene is "on" in one chosen domain (cycling through them)
    # and gets a strong rate boost there. This produces sharp spatial structure
    # aligned with the ground truth domains.
    for g in range(n_patterned):
        on_domain = g % n_domains
        mask = domain_labels == on_domain
        rates[mask, g] += pattern_strength

    # Random genes get a per gene random rate offset that does not depend on
    # position, keeping their marginal level comparable to patterned genes.
    for j in range(n_random):
        g = n_patterned + j
        offset = rng.uniform(0.0, pattern_strength)
        rates[:, g] += offset

    counts = rng.poisson(rates).astype(float)

    gene_names = (
        [f"pat{g:03d}" for g in range(n_patterned)]
        + [f"rnd{j:03d}" for j in range(n_random)]
    )

    return SyntheticTissue(
        counts=counts,
        coords=coords,
        domain_labels=domain_labels,
        gene_is_patterned=gene_is_patterned,
        gene_names=gene_names,
        grid_shape=(rows, cols),
    )
