"""PGSM on the real docs sample: at clock 90 deg and Bx > 0 it must reproduce the served
SWI columns, and the served SWI columns must follow the verified construction."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from space_mango.pgsm import PgsmSpec, imf_sign, to_pgsm

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


@pytest.fixture(scope="module")
def msh() -> pl.DataFrame:
    df = pl.read_parquet(DATA / "magnetosheath", hive_partitioning=True)
    return df.filter(pl.col("Norma_pos") & pl.col("SW_pairing"))


def test_clock_90_reproduces_served_swi(msh):
    out = to_pgsm(msh, PgsmSpec("magnetosheath", cone=(0.0, 89.999), clock=90.0))
    assert out.height > 1000
    for pgsm, swi in [("Bx_pgsm", "Bx_swi"), ("By_pgsm", "By_swi"), ("Bz_pgsm", "Bz_swi"),
                      ("Y_pgsm_norm", "Y_swi_norm"), ("Vz_pgsm", "Vz_swi")]:
        assert np.allclose(out[pgsm].to_numpy(), out[swi].to_numpy(), atol=1e-9)


def test_served_b_swi_follows_the_verified_construction(msh):
    g = lambda *c: msh.select(c).to_numpy()  # noqa: E731
    v, bi, b = g("Vx_sw", "Vy_sw", "Vz_sw"), g("Bx_imf", "By_imf", "Bz_imf"), g("Bx", "By", "Bz")
    s = msh.select(imf_sign()).to_numpy()
    x = -v / np.linalg.norm(v, axis=1)[:, None]
    z = np.cross(x, s * bi)
    z /= np.linalg.norm(z, axis=1)[:, None]
    y = np.cross(z, x)
    rb = s * np.stack([(x * b).sum(1), (y * b).sum(1), (z * b).sum(1)], 1)
    served = g("Bx_swi", "By_swi", "Bz_swi")
    err = np.linalg.norm(rb - served, axis=1) / np.linalg.norm(served, axis=1)
    assert np.mean(err < 1e-6) > 0.99
