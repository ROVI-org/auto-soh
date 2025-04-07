# Performance Tests

This directory contains code used to evaluate the performance of Moirae under different configurations.

## Running a Test

Run a performance test by calling `run-performance-test.py` with arguments
defining the dataset, estimator, and any output levels.
Call `python run-performance-test.py --help` for full details.

The script will append the outcome of the test to `performance-data.jsonl`.

## Adding a New Configuration

Tests are composed of multiple components:

1. *Add a dataset* by including it in the `dataset.yml` registry. 
   Provide a name and a URL to an HDF-format version of the dataset.
2. *Register a model* in the run script. See the mapping between model
   name and Moirae `CellModel` object in the `_model` variable.
3. *Register an estimator* by adding a Python file which creates an 
   OnlineEstimator class for a specific dataset and model to the `online` directory.
   The test script will execute the file and use whichever variable is named `estimator`.
   Place the file in a sub-subdirectory where the first directory is the name
   of the associated dataset and the second is the name of the model.
