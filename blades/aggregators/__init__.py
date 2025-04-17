from .aggregators import Mean, Median, Trimmedmean, GeoMed, DnC
from .centeredclipping import Centeredclipping
from .clippedclustering import Clippedclustering
from .multikrum import Multikrum
from .signguard import Signguard
from .ensemble import EnsembleAggregator

__all__ = classes = [
    "Mean",
    "Median",
    "Trimmedmean",
    "GeoMed",
    "DnC",
    "Clippedclustering",
    "Signguard",
    "Multikrum",
    "Centeredclipping",
    "EnsembleAggregator",
]
