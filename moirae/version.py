"""Tools for recording the version of Moirae"""
from functools import cache
from subprocess import run
from shutil import which
import importlib.metadata

# single source of truth for package version,
# see https://packaging.python.org/en/latest/single_source_version/
__version__ = importlib.metadata.version('moirae')


@cache
def load_git_version() -> str:
    """Load the git version of the git repo

    Returns:
        Git commit summary, if available. ``None`` otherwise.
    """

    if which("git") is None:
        raise ValueError('Cannot find git')
    proc = run('git describe --long --tags --dirty --always'.split(),
               capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f'Git failed: {proc.stderr}')
    return proc.stdout.strip()
