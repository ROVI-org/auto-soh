"""Tools for writing state estimates to HDF5 files"""
from typing import Union, Optional, Literal, Dict, Any, Iterator, Collection
from contextlib import AbstractContextManager, contextmanager
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import pickle as pkl
import json

from flatten_dict import flatten
from pydantic import BaseModel, Field, PrivateAttr
import tables as tb
import pandas as pd
import numpy as np

from moirae import __version__
from moirae.estimators.online import OnlineEstimator, MultivariateRandomDistribution
from moirae.estimators.online.filters.distributions import MultivariateGaussian, DeltaDistribution
from moirae.models.base import HealthVariable, GeneralContainer

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


class HDF5Writer(BaseModel, AbstractContextManager, arbitrary_types_allowed=True):
    """Write state estimation data to an HDF5 file incrementally

    Args:
        hdf5_output: Path to an HDF5 file or group within a file in which to write data
        storage_key: Name of the group within the file to store all states
        table_options: Option used when initializing storage. See :meth:`~pytables.Filters`.
            Default is to use LZF compression.
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
    hdf5_output: Union[Path, str, tb.File] = Field(exclude=True)
    """File or already-open HDF5 file in which to store data"""
    storage_key: str = 'state_estimates'
    """Name of the group in which to store the estimates"""
    table_options: Dict[str, Any] = Field(default_factory=lambda: dict(complib='lzo', complevel=5))
    """Option used when initializing storage. See :class:`~pytables.Filters`"""

    # Attributes defining what is written
    per_timestep: OutputType = 'mean'
    """What information to write each timestep"""
    per_cycle: OutputType = 'full'
    """What information to store at the first timestep each cycle"""

    # State used only while in writing mode
    _file_handle: Optional[tb.File] = PrivateAttr(None)
    """Handle to an open file"""
    _group_handle: Optional[tb.Group] = PrivateAttr(None)
    """Handle to the group being written to"""
    _table_map: Dict[str, tb.Table] = PrivateAttr(default_factory=dict)
    """Map of the currently-open tables"""

    def __enter__(self):
        """Open the file and store the group in which to write data"""
        if not isinstance(self.hdf5_output, tb.File):
            self._file_handle = tb.open_file(self.hdf5_output, mode='a')
        else:
            self._file_handle = self.hdf5_output
        self._group_handle = self._file_handle.create_group('/', self.storage_key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file and stop being ready to write"""
        if not isinstance(self.hdf5_output, tb.File):
            self._file_handle.close()
            self._file_handle = None
        self._group_handle = None
        self._table_map.clear()

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

        Args:
              estimator: Estimator being used to create estimates
              expected_steps: Expected number of estimation timesteps. Required if not :attr:`resizable`.
              expected_cycles: Expected number of cycles.
        """
        self._check_if_ready()

        # Put the metadata in the attributes of the group
        self._group_handle._v_attrs['moirae_version'] = __version__
        self._group_handle._v_attrs['write_settings'] = self.model_dump_json(exclude={'hdf5_output'})

        estimator.get_estimated_state()
        self._group_handle._v_attrs['state_names'] = estimator.state_names
        self._group_handle._v_attrs['output_names'] = estimator.output_names
        self._group_handle._v_attrs['estimator_name'] = estimator.__class__.__name__
        self._group_handle._v_attrs['estimator_pkl'] = pkl.dumps(estimator)
        self._group_handle._v_attrs['distribution_type'] = estimator.state.__class__.__name__
        self._group_handle._v_attrs['cell_model'] = estimator.cell_model.__class__.__name__
        self._group_handle._v_attrs['initial_asoh'] = estimator.asoh.model_dump_json()
        self._group_handle._v_attrs['initial_asoh_pkl'] = pkl.dumps(estimator.asoh)
        self._group_handle._v_attrs['initial_transient_state'] = estimator.transients.model_dump_json()
        self._group_handle._v_attrs['initial_transient_state_pkl'] = pkl.dumps(estimator.transients)

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
            if where == 'per_cycle':
                to_insert['cycle'] = np.array(0, dtype=np.uint32)
            to_insert.update(_convert_state_to_numpy_dict(state, what, 'state'))
            to_insert.update(_convert_state_to_numpy_dict(output, what, 'output'))

            # Make the table
            table_dtype = np.dtype([(k, v.dtype, v.shape) for k, v in to_insert.items()])
            filters = tb.Filters(**self.table_options)
            self._table_map[where] = self._file_handle.create_table(self._group_handle,
                                                                    name=where,
                                                                    description=table_dtype,
                                                                    filters=filters)

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
        for ind, what, where in [(0, self.per_timestep, 'per_timestep'),
                                 (int(cycle), self.per_cycle, 'per_cycle')]:
            # Determine if we must write
            if what == "none":
                continue
            my_table = self._table_map[where]

            # Only write the first state for each cycle
            if where == "per_cycle" and my_table.shape[0] != 0 and my_table[-1]['cycle'] == ind:
                continue

            # Make the row to be inserted
            to_insert = {'time': np.array(time)}
            if where == "per_cycle":
                to_insert['cycle'] = ind
            to_insert.update(_convert_state_to_numpy_dict(state, what, 'state'))
            to_insert.update(_convert_state_to_numpy_dict(output, what, 'output'))

            row = np.empty((1,), dtype=my_table.dtype)
            for k, v in to_insert.items():
                row[k] = v
            my_table.append(row)


@contextmanager
def _open_estimates_hdf5(data_path: Union[str, Path, tb.Group]) -> tb.Group:
    """Access the group storing state estimates in an HDF file,
    regardless of whether someone passes a path or an already-open file"""
    file = None
    if isinstance(data_path, (str, Path)):
        with tb.open_file(data_path, mode='r') as file:
            try:
                yield file.root['state_estimates']
            except ValueError:
                file.close()
                raise
    else:
        yield data_path


def read_state_estimates(
        data_path: Union[str, Path, tb.Group],
        per_timestep: bool = True,
        dist_type: type[MultivariateRandomDistribution] = MultivariateGaussian
) -> Iterator[tuple[float, MultivariateRandomDistribution, MultivariateRandomDistribution]]:
    """
    Read the state estimates from a file into progressive streams

    Args:
        data_path: Path to the HDF5 file or the group holding state estimates from an already-open file.
        per_timestep: Whether to read per-timestep rather than per-cycle estimates
        dist_type: Distribution type to create if the "full" distribution is stored

    Yields:
        - Time associated with estimate
        - Distribution of state estimates
        - Distribution of the output estimates
    """

    # Open the file and access the group holding the state estimates
    with _open_estimates_hdf5(data_path) as se_group:
        # Access the target group to read from
        tag = 'per_timestep' if per_timestep else 'per_cycle'
        read_type = json.loads(se_group._v_attrs['write_settings'])[tag]
        if read_type == 'full':
            if dist_type.__name__ != (stored_type := se_group._v_attrs['distribution_type']):
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
        data_keys = list(data_group.dtype.fields)
        data_keys.remove('time')

        # Yield the data from the file
        def _unpack(row, typ):
            values = dict((k[len(typ):], row[k]) for k in data_keys if k.startswith(typ))

            # Expand variance to covariance
            if read_type == 'mean_var':
                values['covariance'] = np.diag(values.pop('variance'))
            return values

        for i, row in enumerate(data_group):
            # Unpack the dists
            state_values = _unpack(row, 'state_')
            state_dist = dist_type(**state_values)

            output_values = _unpack(row, 'output_')
            output_dist = dist_type(**output_values)
            yield row['time'], state_dist, output_dist


def read_asoh_transient_estimates(data_path: Union[str, Path, tb.Group],
                                  per_timestep: bool = True) \
        -> Iterator[tuple[float, HealthVariable, GeneralContainer]]:
    """
    Read estimates for the battery health and state over time.

    Args:
        data_path: Path to the HDF5 file or the group holding state estimates from an already-open file.
        per_timestep: Whether to read per-timestep rather than per-cycle estimates
    Yields:
         The mean aSOH and battery state at each time step.
    """

    with _open_estimates_hdf5(data_path) as se_group:
        # Get the output types for each
        asoh_template: HealthVariable = se_group._v_attrs['initial_asoh_pkl']
        tran_template: GeneralContainer = se_group._v_attrs['initial_transient_state_pkl']
        num_tran = len(tran_template)

        # Iterate through the estimated states
        for time, state, _ in read_state_estimates(se_group, per_timestep=per_timestep):
            mean = state.get_mean()
            yield time, asoh_template.make_copy(mean[None, num_tran:]), tran_template.make_copy(mean[None, :])


def read_state_estimates_to_df(
        data_path: Union[str, Path, tb.Group],
        per_timestep: bool = True,
        read_std: bool = True,
        read_cov: bool | Collection[tuple[str, str]] = False
) -> pd.DataFrame:
    """
    Read states estimates from the file into a single Pandas DataFrame

    Columns will always include the mean of each estimate and the name
    of the column will start with ``mean_``.

    Columns for the standard deviations start with ``std_``.

    Columns for covariance start are of the form ``cov_(<var1>,<var2>)``.
    The names contain reserved characters for Python and, therefore,
    require using backticks when performing queries in Pandas :meth:`~pandas.DataFrame.query`.

    Args:
        data_path: Path to the HDF5 file or the group holding state estimates from an already-open file.
        per_timestep: Whether to read per-timestep rather than per-cycle estimates
        read_std: Whether to read the standard deviations of each variable
        read_cov: Which subset of covariances to read. Either none (``False``),
            all (``True``), or only a subset provided in a list of pairs

    Returns:
        A pandas dataframe with the requested data
    """

    # Gather the column names
    with _open_estimates_hdf5(data_path) as se_group:
        data = defaultdict(list)

        # Determine the list of covariances to read
        #  Produce a list the column numbers associated with each
        cov_to_read: list[tuple[int, int]] = []
        state_names = se_group._v_attrs['state_names']
        if isinstance(read_cov, bool):
            if read_cov:
                cov_to_read = list(combinations(range(len(state_names)), 2))
        elif isinstance(read_cov, Collection):
            # Check that all names in are in the state names
            all_names = set()
            for names in read_cov:
                all_names.update(names)
            unknown_names = all_names.difference(state_names)
            if len(unknown_names) > 0:
                raise ValueError(f'Covariance requests asks for terms not defined in names: {", ".join(unknown_names)}')

            for st1, st2 in read_cov:
                cov_to_read.append((
                    state_names.index(st1),
                    state_names.index(st2)
                ))
        else:
            raise ValueError(f'Unrecognized argument type for `read_cov`: {type(read_cov)}')

        # Read them
        for time, state, _ in read_state_estimates(se_group, per_timestep):
            data['time'].append(time)

            # Record means
            means = state.get_mean()
            for name, value in zip(state_names, means):
                data[f'mean_{name}'].append(value)

            # Record stds, if desired
            cov = None
            if read_std or len(cov_to_read) > 0:
                cov = state.get_covariance()
            if read_std:
                stds = np.sqrt(np.diag(cov))
                for name, value in zip(state_names, stds):
                    data[f'std_{name}'].append(value)

            # Record covariances
            for id1, id2 in cov_to_read:
                name_1, name_2 = state_names[id1], state_names[id2]
                data[f'cov_({name_1},{name_2})'].append(cov[id1, id2])

    return pd.DataFrame(data)
