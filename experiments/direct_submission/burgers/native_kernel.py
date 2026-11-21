"""ctypes acceleration of the unchanged held-secant exact-memory RK4 update.

The API accepts only current rows, observed-rate memory, an observed secant, and
the declared time/gain constants. It contains no physical model information.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'native_kernel.cpp'
LIBRARY = ROOT/'libnative_kernel.so'
MANIFEST = ROOT/'build_manifest.json'
COMPILE_COMMAND = ['/usr/bin/g++', '-O3', '-std=c++17', '-shared', '-fPIC',
                   '-ffp-contract=off', str(SOURCE), '-o', str(LIBRARY)]
_library = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_kernel():
    """Build once; thereafter verify the saved source/library identities."""
    global _library
    if _library is not None:
        return _library
    if LIBRARY.exists() or MANIFEST.exists():
        if not (LIBRARY.exists() and MANIFEST.exists()):
            raise RuntimeError('partial native build exists; retain it and use a new destination')
        meta = json.loads(MANIFEST.read_text())
        if meta['source_sha256'] != sha(SOURCE) or meta['binary_sha256'] != sha(LIBRARY):
            raise RuntimeError('native source or binary changed after recorded build')
        if meta['compiler_command'] != COMPILE_COMMAND:
            raise RuntimeError('native compiler command differs from recorded build')
    else:
        version = subprocess.run(['/usr/bin/g++', '--version'], check=True,
                                 text=True, capture_output=True).stdout
        compiled = subprocess.run(COMPILE_COMMAND, check=True, text=True, capture_output=True)
        meta = dict(compiler_command=COMPILE_COMMAND, compiler_version=version,
                    compile_stdout=compiled.stdout, compile_stderr=compiled.stderr,
                    source_sha256=sha(SOURCE), binary_sha256=sha(LIBRARY),
                    wrapper_sha256=sha(Path(__file__)), platform=platform.platform(),
                    python=platform.python_version(), numpy=np.__version__,
                    fast_math=False, floating_point_contraction=False,
                    scope='Same RK4/exact held-secant memory/subdivision/retraction; no physics')
        MANIFEST.write_text(json.dumps(meta, indent=2)+'\n')
    native = ctypes.CDLL(str(LIBRARY))
    pointer = ctypes.POINTER(ctypes.c_double)
    native.direct_rows_step.argtypes = [pointer, pointer, pointer, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
    native.direct_rows_step.restype = ctypes.c_int
    _library = native
    return native


def native_step(q, R, v, dt, tau=4., gain=10000., adapt=True):
    """Return (new_q, new_R, subdivisions), leaving supplied arrays unchanged."""
    q = np.array(q, dtype=np.float64, order='C', copy=True)
    R = np.array(R, dtype=np.float64, order='C', copy=True)
    v = np.ascontiguousarray(v, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 5 or R.shape != (len(q), 5, 5) or v.shape != q.shape:
        raise ValueError('expected q/v shape (sites, 5) and R shape (sites, 5, 5)')
    if not len(q) or not np.all(np.isfinite(v)):
        raise ValueError('finite raw-context secants required')
    pointer = ctypes.POINTER(ctypes.c_double)
    steps, message = ctypes.c_int(), ctypes.create_string_buffer(512)
    result = build_kernel().direct_rows_step(q.ctypes.data_as(pointer), R.ctypes.data_as(pointer),
        v.ctypes.data_as(pointer), len(q), dt, tau, gain, int(adapt), ctypes.byref(steps),
        message, len(message))
    if result:
        raise FloatingPointError(message.value.decode())
    return q, R, steps.value


if __name__ == '__main__':
    build_kernel()
    print(MANIFEST.read_text(), end='')
