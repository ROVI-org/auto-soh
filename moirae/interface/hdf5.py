"""Tools for writing state estimates to HDF5 files"""
from contextlib import AbstractContextManager
from typing import Union, Optional, Literal, Dict, Any
from pathlib import Path

from flatten_dict import flatten
from pydantic import BaseModel, Field, PrivateAttr
import numpy as np
import h5py

from moirae.estimators.online import OnlineEstimator, MultivariateRandomDistribution

OutputType = Literal['full', 'mvn', 'mean', 'none']


def _convert_state_to_numpy_dict(state: MultivariateRandomDistribution, what: OutputType) -> Dict[str, np.ndarray]:
    """Convert a multivariate distribution to a dictionary of arrays as requeted by the user."""
    if what == 'full':
        return flatten(state.model_dump(), reducer='dot')
    elif what == 'mvn':
        return {'mean': state.get_mean(), 'covariance': state.get_covariance()}
    elif what == 'mean':
        return {'mean': state.get_mean()}
    else:
        raise ValueError('Mode cannot be none' if what == 'none' else f'Unrecognized what: {what}')


class HDF5Writer(BaseModel, AbstractContextManager, arbitrary_types_allowed=True):
    """Write state estimation data to an HDF5 file incrementally"""

    # Attributes defining where and how to write
    hdf5_output: Union[Path, str, h5py.Group] = Field(exclude=True)
    """File or already-open HDF5 file in which to store data"""
    storage_key: str = 'state_estimates'
    """Name of the group in which to store the estimates"""
    dataset_options: Dict[str, Any] = Field(default_factory=lambda: dict(compression='lzf'))
    """Option used when initializing storage. See :meth:`~h5py.Group.create_dataset`"""
    resizable: bool = True
    """Whether to use `resizable datasets <https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets>`_."""

    # Attributes defining what is written
    per_timestep: OutputType = 'mean'
    """What information to write each timestep"""
    per_cycle: OutputType = 'full'
    """What information to store at the last timestep each cycle"""

    # State used only while in writing mode
    _file_handle: Optional[h5py.File] = PrivateAttr(None)
    """Handle to an open file"""
    _group_handle: Optional[h5py.Group] = PrivateAttr(None)
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
                expected_steps: Optional[int] = None,
                expected_cycles: Optional[int] = None):
        """
        Create the necessary groups and store metadata about the OnlineEstimator

        Additional keyword arguments are passed to :meth:`~h5py.Group.create_dataset`.

        Args:
              estimator: Estimator being used to create estimates
              expected_steps: Expected number of estimation timesteps. Required if not :attr:`resizable`.
              expected_cycles: Expected number of cycles.
        """
        self._check_if_ready()
        if not self.resizable and (expected_steps is None or expected_cycles is None):
            raise ValueError('Expected sizes must be provided if not writing in resizable mode')

        # Put the metadata in the attributes of the group
        self._group_handle.attrs['write_settings'] = self.model_dump_json(exclude={'hdf5_output'})
        self._group_handle.attrs['state_names'] = estimator.state_names
        self._group_handle.attrs['estimator_name'] = estimator.__class__.__name__
        self._group_handle.attrs['cell_model'] = estimator.cell_model.__class__.__name__
        self._group_handle.attrs['initial_asoh'] = estimator.asoh.model_dump_json()
        self._group_handle.attrs['initial_transient_state'] = estimator.transients.model_dump_json()

        # Update accordingly
        state = estimator.state
        for what, where, expected in [(self.per_timestep, 'per_step', expected_steps),
                                      (self.per_cycle, 'per_cycle', expected_cycles)]:
            # Determine what to write
            if what == "none":
                continue
            to_insert = {'time': np.array(0.)}
            to_insert.update(_convert_state_to_numpy_dict(state, what))

            # Create datasets
            my_group = self._group_handle.create_group(where)
            for key, value in to_insert.items():
                if self.resizable:
                    starting_size = 128 if expected is None else expected
                    my_kwargs = {'shape': (starting_size, *value.shape), 'maxshape': (expected, *value.shape)}
                else:
                    my_kwargs = {'shape': (expected, *value.shape)}

                my_group.create_dataset(key, dtype=value.dtype, fillvalue=np.nan, **my_kwargs, **self.dataset_options)

    def write(self, step: int, time: float, cycle: int, state: MultivariateRandomDistribution):
        """
        Write a state estimate to the dataset

        Args:
            step: Index of timestep
            time: Test time of timestep
            cycle: Cycle associated with the timestep
            state: State to be stored
        """
        self._check_if_ready()

        # Write the column to the appropriate part of the HDF5 file
        for ind, what, where in [(step, self.per_timestep, 'per_step'), (cycle, self.per_cycle, 'per_cycle')]:
            # Determine if we must write
            if what == "none":
                continue
            my_group = self._group_handle[where]

            # Only write the first state for each cycle
            if where == "per_cycle" and ind < my_group['time'].shape[0] and not np.isnan(my_group['time'][ind]):
                continue

            # Determine what to write
            to_insert = {'time': np.array(time)}
            to_insert.update(_convert_state_to_numpy_dict(state, what))

            # Write it
            for key, value in to_insert.items():
                my_dataset: h5py.Dataset = my_group[key]

                # Expand by one chunk size if necessary
                if my_dataset.shape[0] <= ind:
                    my_dataset.resize(my_dataset.shape[0] + my_dataset.chunks[0], axis=0)

                my_ind = (ind,) + (slice(None),) * value.ndim
                my_dataset[my_ind] = value
