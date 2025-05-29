"""
Defines capacity extractor
"""
from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle

from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import MaxTheoreticalCapacity


class MaxCapacityExtractor(BaseExtractor):
    """Estimate the maximum discharge capacity of a battery

    Suggested Data: Low current cycles which sample fully charge or discharge a battery

    Algorithm:
        1. Compute the observed capacity each cycle if not available
           using :class:`~battdat.postprocess.integral.CapacityPerCycle`.
        2. Find the maximum capacity over all provided cycles
    """

    def extract(self, data: BatteryDataset) -> MaxTheoreticalCapacity:
        # Access or compute cycle-level capacity
        cycle_stats = data.tables.get('cycle_stats')
        if cycle_stats is None or 'capacity_charge' not in cycle_stats:
            cycle_stats = CapacityPerCycle().compute_features(data)

        max_q = cycle_stats['capacity_charge'].max()
        return MaxTheoreticalCapacity(base_values=max_q)
