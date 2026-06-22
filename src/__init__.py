"""Spatial transcriptomics analysis toolkit.

Synthetic tissue generation, spot clustering, and spatially variable gene
detection built on numpy and scikit-learn.
"""

from .synthetic import SyntheticTissue, make_synthetic_tissue
from .clustering import cluster_spots, match_clusters_to_domains
from .svg import (
    moran_i,
    spatial_weights,
    rank_spatially_variable_genes,
)

__all__ = [
    "SyntheticTissue",
    "make_synthetic_tissue",
    "cluster_spots",
    "match_clusters_to_domains",
    "moran_i",
    "spatial_weights",
    "rank_spatially_variable_genes",
]
