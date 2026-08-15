# -*- coding: utf-8 -*-
"""Sandbox-safe pytest runner.

The DSH file sandbox maps restrictive POSIX modes (mkdir/chmod modes below
0o777) onto Windows ACLs that deny enumeration and removal. pytest creates
its temporary directories with mode 0o700 (basetemp root, pytest-of-<user>,
numbered dirs) and later chmod's them to 0o700, so every such directory
becomes poisoned (PermissionError on scandir, cannot be deleted).

This wrapper forces directory creation modes to 0o777 and widens chmod calls
before running pytest, so tmp_path-based tests work under the sandbox. It is
environment tooling only and is not part of the strategy or its release
checks. On normal Windows hosts the mode argument is ignored anyway, so the
patch does not change test semantics.

Usage:
    python run_pytest_sandbox.py [pytest args...]
"""

import os
import sys

_ORIG_MKDIR = os.mkdir
_ORIG_MAKEDIRS = os.makedirs
_ORIG_CHMOD = os.chmod
_ORIG_OPEN = os.open


def _patched_mkdir(path, mode=0o777):
    return _ORIG_MKDIR(path, 0o777)


def _patched_makedirs(name, mode=0o777, exist_ok=False):
    return _ORIG_MAKEDIRS(name, 0o777, exist_ok=exist_ok)


def _patched_chmod(path, mode):
    return _ORIG_CHMOD(path, mode | 0o777)


def _patched_open(path, flags, mode=0o777, **kwargs):
    return _ORIG_OPEN(path, flags, 0o666, **kwargs)


def _patch():
    os.mkdir = _patched_mkdir
    os.makedirs = _patched_makedirs
    os.chmod = _patched_chmod
    os.open = _patched_open

    import _pytest.pathlib as ppl

    original = ppl.make_numbered_dir

    def sandbox_make_numbered_dir(root, prefix, mode=0o700):
        return original(root, prefix, 0o777)

    ppl.make_numbered_dir = sandbox_make_numbered_dir


def main(argv):
    _patch()
    import pytest

    return pytest.main(argv or ["-q", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
