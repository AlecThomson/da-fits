from __future__ import annotations

import importlib.metadata

import dafits as m


def test_version():
    assert importlib.metadata.version("dafits") == m.__version__
