"""
Unit test for the ECM data checkers.
"""
import pandas as pd
import numpy as np

from pytest import fixture, raises

from moirae.estimators.offline.DataCheckers.base import DataCheckError
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker