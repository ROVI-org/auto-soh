"""Get the maximum capacity of a cell"""
import numpy as np

from moirae.extractors.ecm import MaxCapacityExtractor


def test_max_qt(timeseries_dataset):
    extractor = MaxCapacityExtractor()
    max_cap = extractor.extract(timeseries_dataset)
    assert np.allclose(max_cap.base_values.item(), 10.)
