"""Compatibility shim for HandX NPZ files serialized with NumPy 2.

Install this file as ``sitecustomize.py`` in a NumPy 1 environment. It only
aliases NumPy 2 module names while ``np.load`` is reading an archive.
"""

import sys

import numpy as np


_ORIGINAL_LOAD = np.load
_ORIGINAL_NPZ_GETITEM = np.lib.npyio.NpzFile.__getitem__


def _aliases():
    import numpy.core as numpy_core

    return {
        "numpy._core": numpy_core,
        "numpy._core.multiarray": numpy_core.multiarray,
        "numpy._core.numeric": numpy_core.numeric,
    }


def _with_aliases(callable_):
    aliases = _aliases()
    previous = {
        name: sys.modules.get(name)
        for name in aliases
    }
    sys.modules.update(aliases)
    try:
        return callable_()
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _compatible_load(*args, **kwargs):
    return _with_aliases(lambda: _ORIGINAL_LOAD(*args, **kwargs))


def _compatible_npz_getitem(self, key):
    return _with_aliases(
        lambda: _ORIGINAL_NPZ_GETITEM(self, key)
    )


np.load = _compatible_load
np.lib.npyio.NpzFile.__getitem__ = _compatible_npz_getitem
