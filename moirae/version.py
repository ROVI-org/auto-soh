"""Tools for recording the version of Moirae"""
from functools import cache
from subprocess import run
from shutil import which
import importlib.metadata

# single source of truth for package version,
# see https://packaging.python.org/en/latest/single_source_version/
__version__ = importlib.metadata.version('moirae')
