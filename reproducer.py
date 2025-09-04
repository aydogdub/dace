# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary

NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = (dc.symbol(s, dc.int64)
                                       for s in ('NA', 'NB', 'Nkz', 'NE', 'Nqz', 'Nw', 'Norb', 'N3D'))


@dc.program
def scattering_self_energies_kernel(neigh_idx: dc.int32[NA, NB], dH: dc.complex128[NA, NB, N3D, Norb, Norb],
                                    G: dc.complex128[Nkz, NE, NA, Norb, Norb],
                                    D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D], Sigma: dc.complex128[Nkz, NE, NA, Norb,
                                                                                                      Norb]):

    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD


#### Initialization


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb):
    from numpy.random import default_rng
    rng = default_rng(42)

    neigh_idx = np.ndarray([NA, NB], dtype=np.int32)
    for i in range(NA):
        neigh_idx[i] = np.positive(np.arange(i - NB / 2, i + NB / 2) % NA)
    dH = rng_complex([NA, NB, N3D, Norb, Norb], rng).astype(np.complex128)
    G = rng_complex([Nkz, NE, NA, Norb, Norb], rng).astype(np.complex128)
    D = rng_complex([Nqz, Nw, NA, NB, N3D, N3D], rng).astype(np.complex128)
    Sigma = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)

    return neigh_idx, dH, G, D, Sigma


### Ground Truth


def ground_truth(neigh_idx, dH, G, D, Sigma):

    for k in range(G.shape[0]):
        for E in range(G.shape[1]):
            for q in range(D.shape[0]):
                for w in range(D.shape[1]):
                    for i in range(D.shape[-2]):
                        for j in range(D.shape[-1]):
                            for a in range(neigh_idx.shape[0]):
                                for b in range(neigh_idx.shape[1]):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD



Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb = 2, 4, 2, 2, 2, 6, 2, 3
neigh_idx, dH, G, D, Sigma = initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb)
Sigma_ref = np.copy(Sigma)


# Parse the SDFG and apply auto-opt
sdfg = scattering_self_energies_kernel.to_sdfg()
sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
sdfg.save("s.sdfgz", compress=True)
c = sdfg.compile()
c(neigh_idx=neigh_idx, dH=dH, G=G, D=D, Sigma=Sigma, Nkz=Nkz, NE=NE, Nqz=Nqz, N3D=N3D, NA=NA, NB=NB, Norb=Norb, Nw=Nw)


# Compute ground truth and validate
ground_truth(neigh_idx, dH, G, D, Sigma_ref)
assert np.allclose(Sigma, Sigma_ref), f"Error: The DaCe and ground truth results do not match!"
