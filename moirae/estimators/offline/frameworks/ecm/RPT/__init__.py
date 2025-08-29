"""
Framework for offline estimation of ECM model parameters from RPT data
"""
from typing import Dict, List, Tuple, TypedDict, Union
from typing_extensions import NotRequired, Self

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from battdat.data import BatteryDataset

from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm.advancedSOH import ECMASOH
from moirae.models.ecm.transient import ECMTransientVector
from moirae.models.ecm.components import MaxTheoreticalCapacity, RCComponent

from moirae.estimators.offline.base import BaseOfflineEstimator
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker, FullHPPCDataChecker
from moirae.estimators.offline.DataCheckers.RPT.hppc import FullHPPCCheckerPreinitParams
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.assemblers.utils import SOCRegressor
from moirae.estimators.offline.extractors.ecm import (MaxCapacityCoulEffExtractor,
                                                      R0Extractor,
                                                      RCExtractor,
                                                      OCVExtractor)
from moirae.estimators.offline.extractors.ecm.series_resistance import R0ExtractorPreinitParams
from moirae.estimators.offline.assemblers.ecm import (CapacityAssembler,
                                                      ResistanceAssembler,
                                                      OCVAssembler,
                                                      CapacitanceAssembler)
from moirae.estimators.offline.refiners.loss import MeanSquaredLoss
from moirae.estimators.offline.refiners.scipy import ScipyMinimizer


class ECMAssemblers(TypedDict):
    """
    Auxiliary class to define how ECM-based assemblers are passed
    """
    r0: NotRequired[ResistanceAssembler]
    ocv: NotRequired[OCVAssembler]
    rc: NotRequired[List[Tuple[ResistanceAssembler, CapacitanceAssembler]]]

    @classmethod
    def default(cls) -> Self:
        r0_ass = ResistanceAssembler(regressor=SOCRegressor(style='smooth'))
        ocv_ass = OCVAssembler()
        rc_ass = [(ResistanceAssembler(regressor=SOCRegressor(style='smooth')),
                   CapacitanceAssembler(regressor=SOCRegressor(style='smooth')))]
        return ECMAssemblers(r0=r0_ass, ocv=ocv_ass, rc=rc_ass)


class ECMOfflineEstimatorFromRPT(BaseOfflineEstimator):
    """
    Defines the framework to estimate ECM parameters from RPT data

    Args:
        capacity_checker: Data checker to ensure capacity check cycle meets minimum requirements
        hppc_checker_params: parameters to be used to initialize HPPC data checker once capacity is extracted
        r0_extractor_params: parameters to be used to initialize R0 Extractor once HPPC checker is assembled
    """
    def __init__(self,
                 capacity_checker: CapacityDataChecker = CapacityDataChecker(max_C_rate=0.2),
                 hppc_checker_params: FullHPPCCheckerPreinitParams = FullHPPCCheckerPreinitParams.default(),
                 r0_extractor_params: R0ExtractorPreinitParams = R0ExtractorPreinitParams.default()):
        self.cap_checker = capacity_checker
        self.hppc_checker_params = hppc_checker_params
        self.r0_extractor_params = r0_extractor_params

    @property
    def max_cap_CE_extractor(self) -> MaxCapacityCoulEffExtractor:
        return MaxCapacityCoulEffExtractor(data_checker=self.cap_checker)

    def _assemble_hppc_checker(self,
                               capacity: Union[float, MaxTheoreticalCapacity],
                               coulombic_efficiency: float = 1.) -> FullHPPCDataChecker:
        """
        Helper function to assemble HPPC data checker once capacity and Coulombic efficiency are known
        """
        return FullHPPCDataChecker(capacity=capacity,
                                   coulombic_efficiency=coulombic_efficiency,
                                   **self.hppc_checker_params)

    def extract():
        pass

    def estimate(self,
                 data: Union[pd.DataFrame, BatteryDataset],
                 capacity_check_cycle_number: int,
                 hppc_test_cycle_number: int,
                 number_rc_pairs: int,
                 start_soc: float = 0.0,
                 asoh_assemblers: ECMAssemblers = ECMAssemblers.default(),
                 minimizer_kwargs: Dict = {},
                 *args, **kwargs) -> Tuple[ECMTransientVector, ECMASOH, OptimizeResult]:
        """
        Estimates the aSOH and initial transient states from provided data

        Args:
            data: data to be used for estimation
            capacity_check_cycle_number: cycle number for the capacity check cycle
            hppc_test_cycle_number: cycle number for the HPPC test cycle
            number_rc_pairs: number of RC pairs to be considered
            start_soc: SOC at the beginning of the first diagnostic cycle
            asoh_assemblers: dictionary specifying what assemblers to use for different components of the aSOH
            minimizer_kwargs: keyword arguments to be given to the scipy minimizer

        Returns:
            estimate for aSOH and transient state at the beginning of the diagnostic tests
        """
        # Check for consistency
        if capacity_check_cycle_number == hppc_test_cycle_number:
            raise ValueError('Capacity check cycle number must not coincide with HPPC cycle number!')

        elif abs(capacity_check_cycle_number - hppc_test_cycle_number) != 1:
            raise ValueError('Non-diagnostic cycles present between diagnostic cycles!')

        # Update the assemblers if needed
        use_assemblers = ECMAssemblers.default()
        use_assemblers.update(asoh_assemblers)

        # Ennsure consistency with BatteryDataToolkit
        diagnostic_data = ensure_battery_dataset(data=data)
        diagnostic_raw = diagnostic_data.tables.get('raw_data')
        # Get capacity check and HPPC data
        cap_check_data = diagnostic_raw[diagnostic_raw['cycle_number'] == capacity_check_cycle_number]
        hppc_data = diagnostic_raw[diagnostic_raw['cycle_number'] == hppc_test_cycle_number]

        # Check capacity data
        cap_check_data = self.cap_checker.check(data=cap_check_data)

        # Extract capacity and coulombic efficiency (use `compute_parameters` to skip the data checking)
        qt, ce = self.max_cap_CE_extractor.compute_parameters(data=cap_check_data)
        # Assemble these elements of aSOH
        qt = CapacityAssembler().assemble(extracted_parameter=qt)
        ce = ce['value']

        # Assemble HPPC checker with these, and check data
        hppc_checker = self._assemble_hppc_checker(capacity=qt, coulombic_efficiency=ce)
        hppc_data = hppc_checker.check(data=hppc_data)

        # Before we move on, we need to determine the SOC at the beginning of each cycle
        if capacity_check_cycle_number < hppc_test_cycle_number:
            cap_start_soc = start_soc
            hppc_start_soc = cap_check_data.tables.get('raw_data')['CE_adjusted_charge'].iloc[-1] / qt.amp_hour.item()
            hppc_start_soc += cap_start_soc
        else:
            hppc_start_soc = start_soc
            cap_start_soc = hppc_data.tables.get('raw_data')['CE_adjusted_charge'].iloc[-1] / qt.amp_hour.item()
            cap_start_soc += hppc_start_soc

        # Extract R0 from this
        r0_extractor = R0Extractor(hppc_checker=hppc_checker, **self.r0_extractor_params)
        r0_extracted = r0_extractor.compute_parameters(data=hppc_data, start_soc=hppc_start_soc)
        r0 = use_assemblers['r0'].assemble(extracted_parameter=r0_extracted)

        # With R0, we can get the OCV
        ocv_extractor = OCVExtractor(capacity=qt,
                                     coulombic_efficiency=ce,
                                     series_resistance=r0,
                                     data_checker=self.cap_checker)
        ocv_extracted = ocv_extractor.compute_parameters(data=cap_check_data, start_soc=cap_start_soc)
        ocv = use_assemblers['ocv'].assemble(extracted_parameter=ocv_extracted)

        # Now, if needed, get the RC
        # Prepare list of RC pairs
        rc_pairs = []
        if number_rc_pairs > 0:
            rc_extractor = RCExtractor(rest_checker=hppc_checker.rest_checker)
            rc_extracted = rc_extractor.compute_parameters(data=hppc_data,
                                                           n_rc=number_rc_pairs,
                                                           start_soc=hppc_start_soc)
            # Now, prepare assemblers if needed
            if len(use_assemblers['rc']) == 1:  # if user wishes to use the same assembler for all RC pairs
                use_assemblers['rc'] = use_assemblers['rc'] * number_rc_pairs

            for extracted_tuple, assembler_tuple in zip(rc_extracted, use_assemblers['rc']):
                r = assembler_tuple[0].assemble(extracted_parameter=extracted_tuple[0])
                c = assembler_tuple[1].assemble(extracted_parameter=extracted_tuple[1])
                rc_pairs.append(RCComponent(r=r, c=c))

        # Assemble full aSOH
        asoh = ECMASOH(q_t=qt,
                       ce=ce,
                       ocv=ocv,
                       r0=r0,
                       rc_elements=tuple(rc_pairs))

        # Now, let us refine these estimates
        transient = ECMTransientVector.from_asoh(asoh=asoh)
        transient.soc = np.atleast_2d(start_soc)

        # Remember to make parts of the aSOH updatable!
        asoh.mark_updatable(name='q_t.base_values')
        asoh.mark_updatable(name='r0.base_values')
        asoh.mark_updatable(name='ocv.ocv_ref.base_values')
        for rc in asoh.rc_elements:
            rc.mark_updatable(name='r.base_values')
            rc.mark_updatable(name='c.base_values')
        asoh.mark_updatable(name='h0.base_values')

        # We will limit ourselves to voltage squared-error loss
        loss_metric = MeanSquaredLoss(cell_model=ECM(),
                                      asoh=asoh,
                                      transient_state=transient)
        
        # Prepare minimizer
        minimizer = ScipyMinimizer(objective=loss_metric,
                                   **minimizer_kwargs)

        # We will refine only over the RPT data
        refine_data = diagnostic_raw.query('cycle_number == @capacity_check_cycle_number or '
                                           'cycle_number == @hppc_test_cycle_number')
        refine_data = ensure_battery_dataset(data=refine_data)

        return minimizer.refine(observations=refine_data)
