"""Run a single performance test"""
from argparse import ArgumentParser
from shutil import copyfileobj
from time import perf_counter
from datetime import datetime
from subprocess import run
from platform import node
from pathlib import Path
import logging
import json
import sys
import os

from battdat.data import BatteryDataset
import requests
import yaml

from moirae.interface.hdf5 import HDF5Writer
from moirae.estimators.online import OnlineEstimator
from moirae.models.ecm import EquivalentCircuitModel
from moirae.models.thevenin import TheveninModel
from moirae.interface import run_online_estimate


_models = {
    'ecm': EquivalentCircuitModel(),
    'thevenin': TheveninModel()
}


def load_estimator(text: str, variable_name: str = 'estimator', working_dir: Path | None = None) -> OnlineEstimator:
    """Load an estimator by executing a Python file and retrieving a single variable

    The file should be executed in a directory containing any of the files provided alongside the estimator.

    Args:
        text: Text of a Python file to be executed
        variable_name: Name of variable to be retrieved
        working_dir: Directory in which to execute Python file
    Returns:
        The online estimator implemented in the file
    """

    start_dir = Path.cwd()
    try:
        if working_dir is not None:
            os.chdir(working_dir)
        spec_ns = {}
        exec(text, spec_ns)
        if variable_name not in spec_ns:
            raise ValueError(f'Variable "{variable_name}" not found in')
        return spec_ns[variable_name]
    finally:
        if working_dir is not None:
            os.chdir(start_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    write_levels = ['full', 'mean_cov', 'mean_var', 'mean', 'none']
    parser.add_argument('--dataset', default='camp', help='Dataset used for benchmark test')
    parser.add_argument('--model', default='ecm', help='Numerical model used to describe the battery system')
    parser.add_argument('--estimator', type=Path, required=True, help='Path to the estimator being tested')
    parser.add_argument('--pbar', action='store_true', help='Show the progress bar')
    parser.add_argument('--per-timestep-write', choices=write_levels, default='none',
                        help='How much of the state estimate to write for each timestep')
    parser.add_argument('--per-cycle-write', choices=write_levels, default='none',
                        help='How much of the state estimate to write for each cycle')
    args = parser.parse_args()

    # Start the logger
    my_logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler('run.log', mode='a')]
    for logger in [my_logger, logging.getLogger('moirae')]:
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    my_logger.info('Started performance test')

    # Open the dataset, if not already downloaded
    in_data_path = Path('datasets') / f'{args.dataset}.h5'
    if not in_data_path.exists():
        with open('datasets.yml') as fp:
            datasets = yaml.safe_load(fp)
            if args.dataset not in datasets:
                raise ValueError(f'No such dataset: {args.dataset}. See `datasets.yml` for available options')
            dataset_url = datasets[args.dataset]['url']

        my_logger.info(f'Downloading data for {args.dataset}')
        in_data_path.parent.mkdir(exist_ok=True, parents=True)
        with in_data_path.open('wb') as fp:
            copyfileobj(requests.get(dataset_url, stream=True).raw, fp)
    my_logger.info(f'Using data from: {in_data_path}')

    # Create the numerical model
    if args.model not in _models:
        raise ValueError(f'No such model: {args.model}. Options: {", ".join(_models.keys())}')
    model = _models[args.model]
    my_logger.info(f'Selected model: {args.model}')

    # Get how long to read a file on this filesystem
    start_time = perf_counter()
    BatteryDataset.from_hdf(in_data_path)
    read_time = perf_counter() - start_time
    my_logger.info(f'Reading the dataset from disk requires {read_time:.3f} seconds')

    # Load the model
    if args.estimator.parts[-3:-1] != (args.dataset, args.model):
        my_logger.warning(f'Discrepancy between estimator path {args.estimator} and dataset={args.dataset} model={args.model}')
    estimator = load_estimator(args.estimator.read_text(), working_dir=args.estimator.parent)
    my_logger.info(f'Loaded an {type(estimator)} from {args.estimator}')

    # Make the output writer
    out_file = Path('estimate-outputs.h5')
    est_writer = None
    if args.per_timestep_write != "none" or args.per_cycle_write != "none":
        out_file.unlink(missing_ok=True)
        est_writer = HDF5Writer(hdf5_output=out_file, per_cycle=args.per_cycle_write, per_timestep=args.per_timestep_write)
        my_logger.info(f'Created an HDF5 writer: {est_writer}')

    # Run the estimation
    start_time = perf_counter()
    run_online_estimate(
        dataset=in_data_path,
        estimator=estimator,
        output_states=False,
        pbar=args.pbar,
        hdf5_output=est_writer
    )
    run_time = perf_counter() - start_time
    my_logger.info(f'Estimation completed. Run time {run_time:.2e} s')

    # Store the results
    git_version = run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    output = {
        'hostname': node(),
        'version': git_version,
        'date': datetime.now().isoformat(),
        'model': args.model,
        'dataset': args.dataset,
        'estimator': args.estimator.with_suffix('').name,
        'timestep_write_level': args.per_timestep_write,
        'cycle_write_level': args.per_cycle_write,
        'read_time': read_time,
        'run_time': run_time,
    }
    with open('performance-data.jsonl', 'a') as fp:
        print(json.dumps(output), file=fp)
