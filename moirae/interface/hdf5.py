"""Tools for writing state estimates to HDF5 files"""
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Union, Optional, Literal
from pathlib import Path

from flatten_dict import flatten
import numpy as np
import h5py

from moirae.estimators.online import OnlineEstimator

OutputType = Literal['full', 'mvn', 'mean', 'none']


@dataclass
class HDF5Writer(AbstractContextManager):
    """Write state estimation data to an HDF5 file incrementally"""

    hdf5_output: Union[Path, str, h5py.Group]
    """File or already-open HDF5 file in which to store data"""
    storage_key: str = 'state_estimates'
    """Name of the group in which to store the estimates"""

    _file_handle: Optional[h5py.File] = None
    """Handle to an open file"""
    _group_handle: Optional[h5py.Group] = None
    """Handle to the group being written to"""

    def __enter__(self):
        """Open the file and store the group in which to write data"""
        if not isinstance(self.hdf5_output, h5py.Group):
            root = self._file_handle = h5py.File(self.hdf5_output, mode='a')
        else:
            root = self.hdf5_output
        if self.storage_key not in root:
            root.create_group(self.storage_key)
        self._group_handle = root.get(self.storage_key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file and stop being ready to write"""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._group_handle = None

    @property
    def is_ready(self) -> bool:
        """Whether the class is ready to write estimates"""
        return self._group_handle is not None

    def _check_if_ready(self):
        """Internal: raise exception if class is not ready to write"""
        if not self.is_ready:
            raise ValueError(f'{self.__class__.__name__} is not ready to write. Open it using a `with` statement.')

    def prepare(self,
                estimator: OnlineEstimator,
                expected_size: Optional[int] = None,
                expandable: bool = True,
                what: OutputType = 'full',
                compression: Union[str, int] = 9,
                **kwargs
                ):
        """
        Create the necessary groups and store metadata about the OnlineEstimator

        Additional keyword arguments are passed to :meth:`~h5py.Group.create_dataset`.

        Args:
              estimator: Estimator being used to create estimates
              expected_size: Expected number of rows for the estimates
              expandable: Whether the datasets should be created in expandable mode. Data
              what: Which quantities to save for each timestep
                - "full": The entire estimated state
                - "mvn": A multivariate normal summary (i.e., mean and covariance)
                - "mean": Only the mean estimated value
                - "none": Do not save any information
              compression: Compression argument passed to :meth:`~h5py.Group.create_dataset`.
        """
        self._check_if_ready()
        if expected_size is None and not expandable:
            raise ValueError('Expected size must be provided if not writing in expandable mode')

        # Put the metadata in the attributes of the group
        self._group_handle.attrs['estimator_name'] = estimator.__class__.__name__
        self._group_handle.attrs['cell_whatl'] = estimator.cell_model.__class__.__name__
        self._group_handle.attrs['initial_asoh'] = estimator.asoh.model_dump_json()
        self._group_handle.attrs['initial_transient_state'] = estimator.transients.model_dump_json()

        # Start the list of things to insert
        to_insert = {'time': np.array(0., dtype=np.float64)}

        # Update accordingly
        state = estimator.state
        if what == 'full':
            flattened_state = flatten(state.model_dump(), reducer='dot')
            to_insert.update(flattened_state)
        elif what == 'mvn':
            to_insert['mean'] = state.get_mean()
            to_insert['covariance'] = state.get_covariance()
        elif what == 'mean':
            to_insert['mean'] = state.get_mean()
        else:
            raise ValueError('Mode cannot be none' if what == 'none' else f'Unrecognized what: {what}')

        # Create datasets
        for key, value in to_insert.items():
            if expandable:
                starting_size = 128 if expected_size is None else expected_size
                my_kwargs = {'shape': (starting_size, *value.shape), 'maxshape': (expected_size, *value.shape)}
            else:
                my_kwargs = {'shape': (expected_size, *value.shape)}

            self._group_handle.create_dataset(key, **my_kwargs, dtype=value.dtype, fillvalue=np.nan, **kwargs)
