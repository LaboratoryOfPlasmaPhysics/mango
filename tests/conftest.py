"""Shared fixtures: a small Hive-partitioned dataset with the schema the live server serves.

SERVED_COLUMNS is copied from GET /api/v1/regions/{r}/columns on the public server
(2026-10-07). Do NOT derive it from space_mango.models: it is the ground truth the
catalog is tested against.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from space_mango.app import create_app
from space_mango.client import MangoClient
from space_mango.dataset import MangoDataset, get_dataset

Row = dict[str, object]

_BASE = ["Time", "Bx", "By", "Bz", "Np", "Vx", "Vy", "Vz", "Tp", "X_gsm", "Y_gsm", "Z_gsm"]
_PAIRED = [
    "SW_pairing", "Bx_imf", "By_imf", "Bz_imf", "Np_sw", "Vx_sw", "Vy_sw", "Vz_sw",
    "Tp_sw", "Pd_sw", "Beta_sw", "Ma_sw", "R_mp", "R_norm", "Norma_pos",
    "X_gsm_norm", "Y_gsm_norm", "Z_gsm_norm",
]
_SWI = [
    "R_bs", "Bx_swi", "By_swi", "Bz_swi", "Vx_swi", "Vy_swi", "Vz_swi",
    "X_swi_norm", "Y_swi_norm", "Z_swi_norm",
]
# "SC" is not listed: it comes from the SC=<name> directory (Hive partition key).
SERVED_COLUMNS: dict[str, list[str]] = {
    "solar_wind": _BASE,
    "magnetosphere": _BASE + _PAIRED + ["tilt"],
    "magnetosheath": _BASE + _PAIRED + _SWI,
}
_BOOL_COLUMNS = {"SW_pairing", "Norma_pos"}


def _make_row(region: str, sc: str, time: datetime, **values: object) -> Row:
    cols = SERVED_COLUMNS[region]
    unknown = set(values) - set(cols)
    if unknown:
        raise KeyError(f"{sorted(unknown)} are not served in region {region!r}")
    row: Row = {c: (True if c in _BOOL_COLUMNS else 1.0) for c in cols}
    row["Time"] = time
    row["SC"] = sc
    row.update(values)
    return row


def _write_region(base: Path, region: str, rows: list[Row]) -> None:
    df = pl.DataFrame(rows).with_columns(pl.col("Time").cast(pl.Datetime("ns")))
    for sc in df["SC"].unique().to_list():
        sc_dir = base / region / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("SC") == sc).drop("SC").write_parquet(sc_dir / "part-0.parquet")


def _write_dataset(base: Path, rows_by_region: dict[str, list[Row]]) -> Path:
    for region, rows in rows_by_region.items():
        _write_region(base, region, rows)
    return base


def _api(data_dir: Path) -> TestClient:
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    return TestClient(app)


def _client(data_dir: Path) -> MangoClient:
    tc = _api(data_dir)
    return MangoClient(
        "http://testserver",
        transport=tc._transport,
        cache_dir=data_dir.parent / f"{data_dir.name}-cache",
    )


def standard_rows() -> dict[str, list[Row]]:
    r = _make_row
    msh = "magnetosheath"
    msp = "magnetosphere"
    return {
        msh: [
            r(msh, "THA", datetime(2016, 3, 15, 10), Bx=1.0, By=2.0, Bz=3.0, Np=10.0,
              Vx=-200.0, Vy=0.0, Vz=0.0, Tp=1e6, X_gsm=8.0, Y_gsm=3.0, Z_gsm=0.0,
              R_norm=0.4, SW_pairing=True, Bz_imf=-5.0, By_imf=1.0, Bx_imf=0.5,
              Pd_sw=4.0, Np_sw=8.0, Tp_sw=1e5, Vx_sw=-400.0, Beta_sw=1.2, Ma_sw=7.0,
              Norma_pos=True),
            r(msh, "MMS", datetime(2018, 7, 20, 14, 30), Bx=2.0, By=3.0, Bz=4.0, Np=20.0,
              Vx=-300.0, Vy=1.0, Vz=1.0, Tp=2e6, X_gsm=9.0, Y_gsm=4.0, Z_gsm=1.0,
              R_norm=0.8, SW_pairing=False, Bz_imf=2.0, By_imf=-1.0, Bx_imf=-0.3,
              Pd_sw=1.0, Np_sw=5.0, Tp_sw=5e4, Vx_sw=-350.0, Beta_sw=0.8, Ma_sw=5.0,
              Norma_pos=False),
            r(msh, "C1", datetime(2019, 1, 5, 8), Bx=0.5, By=-1.0, Bz=-2.0, Np=15.0,
              Vx=-250.0, Vy=-0.5, Vz=0.5, Tp=1.5e6, X_gsm=10.0, Y_gsm=-2.0, Z_gsm=0.5,
              R_norm=0.2, SW_pairing=True, Bz_imf=-8.0, By_imf=3.0, Bx_imf=1.0,
              Pd_sw=6.0, Np_sw=12.0, Tp_sw=2e5, Vx_sw=-500.0, Beta_sw=2.0, Ma_sw=10.0,
              Norma_pos=True),
        ],
        msp: [
            r(msp, "MMS", datetime(2015, 6, 10, 12), Bx=10.0, By=-5.0, Bz=-20.0, Np=1.0,
              Vx=-50.0, Vy=10.0, Vz=5.0, Tp=5e7, X_gsm=-5.0, Y_gsm=2.0, Z_gsm=1.0,
              R_norm=0.3, tilt=0.15, SW_pairing=True, Bz_imf=-3.0, By_imf=0.0, Bx_imf=0.0,
              Pd_sw=2.0, Np_sw=6.0, Tp_sw=1e5, Vx_sw=-380.0, Beta_sw=1.0, Ma_sw=6.0,
              Norma_pos=True),
            r(msp, "THA", datetime(2017, 11, 3, 6), Bx=15.0, By=3.0, Bz=-30.0, Np=0.5,
              Vx=-30.0, Vy=5.0, Vz=-2.0, Tp=8e7, X_gsm=-8.0, Y_gsm=-1.0, Z_gsm=-0.5,
              R_norm=0.6, tilt=-0.1, SW_pairing=True, Bz_imf=1.0, By_imf=2.0, Bx_imf=-1.0,
              Pd_sw=3.0, Np_sw=7.0, Tp_sw=1.5e5, Vx_sw=-420.0, Beta_sw=1.5, Ma_sw=8.0,
              Norma_pos=True),
            r(msp, "C3", datetime(2021, 2, 14, 18), Bx=5.0, By=-2.0, Bz=-10.0, Np=2.0,
              Vx=-80.0, Vy=0.0, Vz=0.0, Tp=3e7, X_gsm=-3.0, Y_gsm=5.0, Z_gsm=2.0,
              R_norm=0.9, tilt=0.05, SW_pairing=False, Bz_imf=-1.0, By_imf=-3.0, Bx_imf=0.5,
              Pd_sw=1.5, Np_sw=4.0, Tp_sw=8e4, Vx_sw=-360.0, Beta_sw=0.6, Ma_sw=4.0,
              Norma_pos=False),
        ],
        "solar_wind": [
            r("solar_wind", "THA", datetime(2016, 5, 1), Bx=0.1, By=-0.5, Bz=-1.0, Np=5.0,
              Vx=-400.0, Vy=0.0, Vz=0.0, Tp=1e5, X_gsm=20.0, Y_gsm=0.0, Z_gsm=0.0),
        ],
    }


@pytest.fixture
def make_row() -> Callable[..., Row]:
    return _make_row


@pytest.fixture
def make_dataset(tmp_path: Path) -> Callable[[dict[str, list[Row]]], Path]:
    counter = iter(range(1_000))

    def factory(rows_by_region: dict[str, list[Row]]) -> Path:
        return _write_dataset(tmp_path / f"data{next(counter)}", rows_by_region)

    return factory


@pytest.fixture
def make_api() -> Callable[[Path], TestClient]:
    return _api


@pytest.fixture
def make_client() -> Callable[[Path], MangoClient]:
    return _client


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("mango_data"), standard_rows())


@pytest.fixture(scope="session")
def api(dataset_dir: Path) -> TestClient:
    return _api(dataset_dir)


@pytest.fixture(scope="session")
def client(dataset_dir: Path) -> MangoClient:
    return _client(dataset_dir)
