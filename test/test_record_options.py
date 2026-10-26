"""Tests for the record_state memory options (spec O1-O5).

Covers record_arrays selection, record_dtype=float32, record_to memmap
backing with metadata + Model.load_state, record_every subsampling, and
the removal of the dead Jacobian placeholder allocations.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model


def make_df(n=60, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    df["J"] = 1.0
    df["Q"] = 1.0
    df["S_0"] = 5.0
    df["C"] = rng.standard_normal(n)
    return df


SPECS = {"Q": {"Q_SAS": {"ST": [0, 5.0]}}}
SOLS = {"C": {"C_old": 1.0}}


def run_model(df=None, **kwargs):
    m = Model(
        df if df is not None else make_df(),
        sas_specs=SPECS,
        solute_parameters=SOLS,
        dt=0.1,
        record_state=True,
        **kwargs,
    )
    m.run()
    return m


@pytest.fixture(scope="module")
def full_run():
    return run_model()


class TestRecordArrays:
    def test_subset_matches_full_run(self, full_run):
        m = run_model(record_arrays={"sT", "pQ"})
        np.testing.assert_allclose(np.asarray(m.result["sT"]), np.asarray(full_run.result["sT"]))
        np.testing.assert_allclose(np.asarray(m.result["pQ"]), np.asarray(full_run.result["pQ"]))

    def test_unselected_arrays_absent_with_helpful_error(self):
        m = run_model(record_arrays={"sT", "pQ"})
        assert "mQ" not in m.result
        with pytest.raises(KeyError, match="not recorded"):
            m.result["mQ"]
        with pytest.raises(KeyError, match="not recorded"):
            m.result["solute_balance"]

    def test_c_q_unaffected_by_selection(self, full_run):
        m = run_model(record_arrays={"sT"})
        np.testing.assert_allclose(np.asarray(m.result["C_Q"]), np.asarray(full_run.result["C_Q"]))

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown record_arrays"):
            run_model(record_arrays={"sT", "bogus"})

    def test_balance_dependency_enforced(self):
        with pytest.raises(ValueError, match="water_balance.*requires"):
            run_model(record_arrays={"water_balance"})
        with pytest.raises(ValueError, match="solute_balance.*requires"):
            run_model(record_arrays={"solute_balance", "mT"})

    def test_string_other_than_all_rejected(self):
        with pytest.raises(TypeError, match="record_arrays"):
            run_model(record_arrays="sT")

    def test_no_solutes_drops_solute_arrays(self):
        df = make_df()
        m = Model(df, sas_specs=SPECS, dt=0.1, record_state=True)
        m.run()
        assert "sT" in m.result and "water_balance" in m.result
        assert "mT" not in m.result and "solute_balance" not in m.result


class TestRecordDtype:
    def test_float32_close_to_float64(self, full_run):
        m = run_model(record_dtype="float32")
        assert np.asarray(m.result["sT"]).dtype == np.float32
        for name in ("sT", "pQ", "mT", "mQ"):
            a64 = np.asarray(full_run.result[name], dtype=np.float64)
            a32 = np.asarray(m.result[name], dtype=np.float64)
            np.testing.assert_allclose(a32, a64, rtol=2e-6, atol=1e-6)

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="record_dtype"):
            run_model(record_dtype="float16")

    def test_float32_state_ok_under_evapoconcentration(self):
        """Q3 experiment as regression guard: float32 state arrays stay at
        float32-epsilon relative accuracy even with strong evapoconcentration
        and a wide concentration range; balance diagnostics degrade (documented)."""
        n = 200
        rng = np.random.default_rng(7)
        df = pd.DataFrame(index=range(n))
        df["J"] = 2.0
        df["Q"] = 1.0
        df["E"] = 1.0
        df["S_0"] = 0.5
        df["S_m"] = 0.0
        df["C"] = 10.0 ** rng.uniform(-3, 3, n)
        specs = {
            "Q": {"spec": {"ST": ["S_m", "S_0"]}},
            "E": {"spec": {"ST": ["S_m", "S_0"]}},
        }
        sols = {"C": {"C_old": 1e-3, "alpha": {"Q": 1.0, "E": 0.0}}}

        results = {}
        for dtype in ("float64", "float32"):
            m = Model(
                df.copy(),
                sas_specs=specs,
                solute_parameters=sols,
                dt=0.1,
                n_substeps=2,
                record_state=True,
                record_dtype=dtype,
            )
            m.run()
            results[dtype] = m
        for name in ("sT", "mT", "mQ"):
            a64 = np.asarray(results["float64"].result[name], dtype=np.float64)
            a32 = np.asarray(results["float32"].result[name], dtype=np.float64)
            scale = np.abs(a64)
            mask = scale > scale.max() * 1e-12
            rel = np.max(np.abs(a32[mask] - a64[mask]) / scale[mask])
            assert rel < 1e-5, f"{name}: float32 rel err {rel:.2e}"


class TestRecordTo:
    def test_memmap_equivalence_and_files(self, full_run, tmp_path):
        d = str(tmp_path / "run1")
        m = run_model(record_to=d)
        for name in ("sT", "pQ", "mT", "mQ", "mR", "water_balance", "solute_balance"):
            assert os.path.exists(os.path.join(d, name + ".npy"))
            np.testing.assert_allclose(np.asarray(m.result[name]), np.asarray(full_run.result[name]))
        # Results are read-only memmaps
        assert isinstance(m.result["sT"], np.memmap)
        with pytest.raises(ValueError):
            m.result["sT"][0, 0] = 99.0

    def test_metadata_written(self, tmp_path):
        d = str(tmp_path / "run2")
        run_model(record_to=d, record_dtype="float32")
        meta = json.load(open(os.path.join(d, "mesas_run.json")))
        assert meta["format"] == "mesas-run-v1"
        assert meta["record_dtype"] == "float32"
        assert meta["fluxorder"] == ["Q"]
        assert meta["solorder"] == ["C"]
        assert set(meta["record_arrays"]) >= {"sT", "pQ", "mQ"}
        assert os.path.exists(os.path.join(d, "index_ts.npy"))
        assert os.path.exists(os.path.join(d, "C_Q.npy"))

    def test_load_state_round_trip(self, full_run, tmp_path):
        d = str(tmp_path / "run3")
        run_model(record_to=d)
        r = Model.load_state(d)
        np.testing.assert_allclose(np.asarray(r["sT"]), np.asarray(full_run.result["sT"]))
        np.testing.assert_allclose(np.asarray(r["C_Q"]), np.asarray(full_run.result["C_Q"]))
        assert r["run_metadata"]["timeseries_length"] == 60
        assert len(np.asarray(r["index_ts"])) == 60

    def test_load_state_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mesas_run.json"):
            Model.load_state(str(tmp_path / "nope"))

    def test_memmap_with_subset(self, tmp_path):
        d = str(tmp_path / "run4")
        m = run_model(record_to=d, record_arrays={"sT", "pQ"})
        assert os.path.exists(os.path.join(d, "sT.npy"))
        assert not os.path.exists(os.path.join(d, "mQ.npy"))
        r = Model.load_state(d)
        np.testing.assert_allclose(np.asarray(r["sT"]), np.asarray(m.result["sT"]))


class TestRecordEvery:
    def test_shapes_and_values(self, full_run):
        k = 10
        m = Model(
            make_df(),
            sas_specs=SPECS,
            solute_parameters=SOLS,
            dt=0.1,
            record_every=k,
        )
        m.run()
        n_rec = 60 // k
        assert m.result["sT"].shape == (60, n_rec + 1)
        # Recorded step i corresponds to full-run step k-1, 2k-1, ...
        full_sT = np.asarray(full_run.result["sT"])
        sub_sT = np.asarray(m.result["sT"])
        for i in range(n_rec):
            np.testing.assert_allclose(sub_sT[:, i + 1], full_sT[:, (i + 1) * k])

    def test_validation(self):
        with pytest.raises(ValueError, match="positive integer"):
            run_model(record_every=0)
        df = make_df()
        df["flag"] = df.index % 7 == 0
        with pytest.raises(ValueError, match="record_state column"):
            m = Model(
                df,
                sas_specs=SPECS,
                solute_parameters=SOLS,
                dt=0.1,
                record_state="flag",
                record_every=5,
            )
            m.run()


class TestPlaceholderRemoval:
    def test_jacobian_placeholders_are_singletons(self, full_run):
        """O1: the unused Jacobian outputs must not be allocated at O(N^2)."""
        assert np.asarray(full_run.result["dsTdSj"]).size == 1
        assert np.asarray(full_run.result["dmTdSj"]).size == 1
        assert np.asarray(full_run.result["dCdSj"]).size == 1

    def test_get_jacobian_raises_clearly(self, full_run):
        with pytest.raises(NotImplementedError, match="numerical"):
            full_run.get_jacobian()


class TestBalancesStillClose:
    def test_machine_precision_balances_float64(self, full_run):
        assert np.abs(np.asarray(full_run.result["water_balance"])).max() < 1e-12
        assert np.abs(np.asarray(full_run.result["solute_balance"])).max() < 1e-12
