"""Tools for writing state estimates to HDF5 files"""
from contextlib import AbstractContextManager
from typing import Union, Optional, Literal, Dict, Any, Iterator
from pathlib import Path
import json

from flatten_dict import flatten
from pydantic import BaseModel, Field, PrivateAttr
import numpy as np
import h5py

from moirae.estimators.online import OnlineEstimator, MultivariateRandomDistribution
from moirae.estimators.online.filters.distributions import MultivariateGaussian, DeltaDistribution

OutputType = Literal['full', 'mean_cov', 'mean_var', 'mean', 'none']


def _convert_state_to_numpy_dict(state: MultivariateRandomDistribution,
                                 what: OutputType, tag: str) -> Dict[str, np.ndarray]:
    """Convert a multivariate distribution to a dictionary of arrays as requested by the user.

    Args:
        state: State to be stored
        what: What to store
        tag: Name of the distribution (e.g., state, outputs)
    """
    if what == 'full':
        return dict((f'{tag}_{k}', v) for k, v in flatten(state.model_dump(), reducer='dot').items())
    elif what == 'mean_cov':
        return {f'{tag}_mean': state.get_mean(), f'{tag}_covariance': state.get_covariance()}
    elif what == 'mean_var':
        return {f'{tag}_mean': state.get_mean(), f'{tag}_variance': np.diag(state.get_covariance())}
    elif what == 'mean':
        return {f'{tag}_mean': state.get_mean()}
    else:
        raise ValueError('Mode cannot be none' if what == 'none' else f'Unrecognized what: {what}')


# TODO (wardlt): Consider writing only every N timesteps
class HDF5Writer(BaseModel, AbstractContextManager, arbitrary_types_allowed=True):
    """Write state estimation data to an HDF5 file incrementally

    Args:
        hdf5_output: Path to an HDF5 file or group within a file in which to write data
        storage_key: Name of the group within the file to store all states
        dataset_options: Option used when initializing storage. See :meth:`~h5py.Group.create_dataset`.
            Default is to use LZF compression.
        resizable: Whether to allow the file to be shrunk or expanded
        per_timestep: Which information to store at each timestep:
            - `full`: All available information about the estimated state
            - `mean_cov`: The mean and covariance of the estimated state
            - `mean_var`: The mean and variance (i.e., diagonal of covariance matrix) of the estimated state
            - `mean`: Only the mean
            - `none`: No information
        per_cycle: Which information to write at the first step of a cycle. The options are the same
            as `per_timestep`.
    """

    # Attributes defining where and how to write
    hdf5_output: Union[Path, str, h5py.Group] = Field(exclude=True)
    """File or already-open HDF5 file in which to store data"""
    file_options: Dict[str, Any] = Field(default_factory=lambda: dict(rdcc_nbytes=1024 ** 2 * 16, rdcc_w0=0))
    """Options employed when opening the HDFt file. See :meth:`~h5py.File`"""
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
    """What information to store at the first timestep each cycle"""

    # State used only while in writing mode
    _file_handle: Optional[h5py.File] = PrivateAttr(None)
    """Handle to an open file"""
    _group_handle: Optional[h5py.Group] = PrivateAttr(None)
    """Handle to the group being written to"""
    position: Optional[int] = None
    """Index of the next step to be written"""

    def __enter__(self):
        """Open the file and store the group in which to write data"""
        if not isinstance(self.hdf5_output, h5py.Group):
            root = self._file_handle = h5py.File(self.hdf5_output, mode='a', **self.file_options)
        else:
            root = self.hdf5_output
        if self.storage_key not in root:
            root.create_group(self.storage_key)
        self._group_handle = root.get(self.storage_key)
        self.position = 0  # TODO (wardlt): Support re-opening files for writing phase
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
        self._group_handle.attrs['output_names'] = estimator.output_names
        self._group_handle.attrs['estimator_name'] = estimator.__class__.__name__
        self._group_handle.attrs['distribution_type'] = estimator.state.__class__.__name__
        self._group_handle.attrs['cell_model'] = estimator.cell_model.__class__.__name__
        self._group_handle.attrs['initial_asoh'] = estimator.asoh.model_dump_json()
        self._group_handle.attrs['initial_transient_state'] = estimator.transients.model_dump_json()

        # Update accordingly
        state = estimator.state
        num_outputs = estimator.num_output_dimensions
        # TODO (wardlt): Use actual output class employed by estimator
        output = MultivariateGaussian(mean=np.zeros((num_outputs,)),
                                      covariance=np.zeros((num_outputs, num_outputs)))

        for what, where, expected in [(self.per_timestep, 'per_timestep', expected_steps),
                                      (self.per_cycle, 'per_cycle', expected_cycles)]:
            # Determine what to write
            if what == "none":
                continue
            to_insert = {'time': np.array(0.)}
            to_insert.update(_convert_state_to_numpy_dict(state, what, 'state'))
            to_insert.update(_convert_state_to_numpy_dict(output, what, 'output'))

            # Create datasets
            if where in self._group_handle:
                raise ValueError(f'File contains {self.storage_key}/{where} group. Overwriting not yet supported')
            my_group = self._group_handle.create_group(where)
            for key, value in to_insert.items():
                if self.resizable:
                    starting_size = 128 if expected is None else expected
                    my_kwargs = {'shape': (starting_size, *value.shape), 'maxshape': (expected, *value.shape)}
                else:
                    my_kwargs = {'shape': (expected, *value.shape)}

                my_group.create_dataset(key, dtype=value.dtype, fillvalue=np.nan, **my_kwargs, **self.dataset_options)

    def append_step(self,
                    time: float,
                    cycle: int,
                    state: MultivariateRandomDistribution,
                    output: MultivariateRandomDistribution):
        """
        Add a state estimate to the dataset

        Args:
            time: Test time of timestep
            cycle: Cycle associated with the timestep
            state: State to be stored
            output: Outputs predicted from the estimator
        """
        self._check_if_ready()

        # Write the column to the appropriate part of the HDF5 file
        # TODO (wardlt): Introduce a batched write implementation.
        for ind, what, where in [(self.position, self.per_timestep, 'per_timestep'),
                                 (int(cycle), self.per_cycle, 'per_cycle')]:
            # Determine if we must write
            if what == "none":
                continue
            my_group = self._group_handle[where]

            # Only write the first state for each cycle
            if where == "per_cycle" and ind < my_group['time'].shape[0] and not np.isnan(my_group['time'][ind]):
                continue

            # Determine what to write
            to_insert = {'time': np.array(time)}
            to_insert.update(_convert_state_to_numpy_dict(state, what, 'state'))
            to_insert.update(_convert_state_to_numpy_dict(output, what, 'output'))

            # Write it
            for key, value in to_insert.items():
                my_dataset: h5py.Dataset = my_group[key]

                # Expand by one chunk size if necessary
                if my_dataset.shape[0] <= ind:
                    my_dataset.resize(my_dataset.shape[0] + my_dataset.chunks[0], axis=0)

                my_ind = (ind,) + (slice(None),) * value.ndim
                my_dataset[my_ind] = value

        # Increment the step position
        self.position += 1


def read_state_estimates(data_path: Union[str, Path, h5py.Group],
                         per_timestep: bool = True,
                         dist_type: type[MultivariateRandomDistribution] = MultivariateGaussian
                         ) -> Iterator[tuple[float, MultivariateRandomDistribution, MultivariateRandomDistribution]]:
    """
    Read the state estimates from a file into progressive streams

    Args:
        data_path: Path to the HDF5 file or the group holding state estimates from an already-open file.
        per_timestep: Whether to read per-timestep rather than per-cycle estiamtes
        dist_type: Distribution type to create if the "full" distribution is stored

    Yields:
        - Time associated with estimate
        - Distribution of state estimates
        - Distribution of the output estimates
    """

    # Open the file and access the group holding the state estimates
    file = None
    if isinstance(data_path, (str, Path)):
        file = h5py.File(data_path, mode='r')
        try:
            se_group = file['state_estimates']
        except ValueError:
            file.close()
            raise
    else:
        se_group = data_path

    try:
        # Access the target group to read from
        tag = 'per_timestep' if per_timestep else 'per_cycle'
        read_type = json.loads(se_group.attrs['write_settings'])[tag]
        if read_type == 'full':
            if dist_type.__name__ != (stored_type := se_group.attrs['distribution_type']):
                # TODO (wardlt): Have Moriae load the correct one automatically
                raise ValueError(f'Estimates were stored as a {stored_type}'
                                 f' but you provided dist_type={dist_type.__name__}')
        elif read_type in ['mean_cov', 'mean_var']:
            dist_type = MultivariateGaussian
        elif read_type == 'mean':
            dist_type = DeltaDistribution
        elif read_type == 'none':
            raise ValueError(f'No data was written for {tag}')
        else:
            raise NotImplementedError(f'Support for {read_type} not yet implemented')
        data_group = se_group[tag]

        # Read what data are available
        data_keys = list(data_group.keys())
        data_keys.remove('time')

        # Yield the data from the file
        def _unpack(ind, typ):
            values = dict((k[len(typ):], data_group[k][ind]) for k in data_keys if k.startswith(typ))

            # Expand variance to covariance
            if read_type == 'mean_var':
                values['covariance'] = np.diag(values.pop('variance'))
            return values

        for i, time in enumerate(data_group['time']):
            if np.isnan(time):  # NaN marks the end of the data
                return
            # Unpack the dists
            state_values = _unpack(i, 'state_')
            state_dist = dist_type(**state_values)

            output_values = _unpack(i, 'output_')
            output_dist = dist_type(**output_values)
            yield time, state_dist, output_dist
    finally:
        if file is not None:
            file.close()
