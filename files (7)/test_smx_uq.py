import sys, os, json, tempfile, unittest, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from smx_uq import calibration as C, intervals as I, risk as R, ood as O, explain as E, pipeline as P, maps as M
from harness import make_synth, make_ctx


class TestCalibration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 3000
        z = rng.normal(0, 1.5, self.n)
        self.y = (rng.uniform(size=self.n) < 1 / (1 + np.exp(-z))).astype(int)
        self.raw = 1 / (1 + np.exp(-(2.5 * z + 1)))            # over-confident and shifted
        self.g = rng.integers(0, 12, self.n).astype(str)

    def test_sigmoid_and_isotonic_improve_logloss(self):
        base = C.calibration_report(self.y, self.raw)["LogLoss"]
        for m in ("sigmoid", "isotonic"):
            q = C.crossfit_calibrated(self.raw, self.y, self.g, m)
            self.assertLess(C.calibration_report(self.y, q)["LogLoss"], base)

    def test_outputs_valid_probabilities_and_nan_preserved(self):
        for m in ("none", "sigmoid", "isotonic"):
            c = C.Calibrator(m).fit(self.raw, self.y)
            p = np.r_[self.raw[:5], np.nan]
            out = c.predict(p)
            self.assertTrue(np.isnan(out[-1])); self.assertTrue(np.all((out[:-1] >= 0) & (out[:-1] <= 1)))

    def test_isotonic_monotone(self):
        c = C.Calibrator("isotonic").fit(self.raw, self.y)
        x = np.linspace(0, 1, 200)
        self.assertTrue(np.all(np.diff(c.predict(x)) >= -1e-12))

    def test_selection_ineligible_isotonic_with_few_positives(self):
        idx = np.where(self.y == 1)[0][:10].tolist() + np.where(self.y == 0)[0][:200].tolist()
        m, rows = C.select_method(self.raw[idx], self.y[idx], self.g[idx], min_pos_isotonic=40)
        self.assertNotEqual(m, "isotonic"); self.assertFalse([r for r in rows if r["method"] == "isotonic"][0]["eligible"])

    def test_selection_keeps_none_when_already_calibrated(self):
        p = 1 / (1 + np.exp(-np.random.default_rng(1).normal(0, 1.5, 4000)))
        y = (np.random.default_rng(2).uniform(size=4000) < p).astype(int)
        m, _ = C.select_method(p, y, np.random.default_rng(3).integers(0, 10, 4000).astype(str), min_gain=0.002)
        self.assertEqual(m, "none")

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            C.Calibrator("sigmoid").fit(self.raw[:50], np.zeros(50))

    def test_crossfit_never_shares_groups(self):
        # a calibrator trained on fold k must not see group g_k: verify via equality with manual leave-groups-out
        from sklearn.model_selection import GroupKFold
        out = C.crossfit_calibrated(self.raw, self.y, self.g, "sigmoid", 4)
        for tr, te in GroupKFold(4).split(self.raw, self.y, self.g):
            self.assertFalse(set(self.g[tr]) & set(self.g[te]))
            ref = C.Calibrator("sigmoid").fit(self.raw[tr], self.y[tr]).predict(self.raw[te])
            np.testing.assert_allclose(out[te], ref)


class TestIntervals(unittest.TestCase):
    def data(self, n, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, n)
        sd = 0.1 + 0.6 * x
        y = 1 + x + rng.normal(0, sd)
        return pd.DataFrame({"x": x}), np.clip(y, 0, None), 1 + x, sd

    def make_ci(self, scale):
        Xtr, ytr, mtr, _ = self.data(3000, 1)
        resid = np.abs(ytr - mtr)
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer
        prep = lambda: ColumnTransformer([("n", FunctionTransformer(), ["x"])])
        ci = I.ConformalInterval(0.1, scale).fit_scale(prep, Xtr, resid)
        Xc, yc, mc, _ = self.data(1500, 2)
        ci.calibrate(Xc, yc, mc)
        return ci

    def test_coverage_near_nominal_and_widths_adapt(self):
        for scale in ("normalized", "absolute"):
            ci = self.make_ci(scale)
            Xt, yt, mt, _ = self.data(5000, 3)
            iv = ci.predict_interval(Xt, mt)
            cov = ((yt >= iv.lo_log) & (yt <= iv.hi_log)).mean()
            self.assertAlmostEqual(cov, 0.9, delta=0.03, msg=scale)
        iv = self.make_ci("normalized").predict_interval(*self.data(500, 4)[:1], self.data(500, 4)[2])
        self.assertGreater(np.corrcoef(iv.sigma_log, iv.width_log)[0, 1], 0.9)     # heteroscedastic widths
        iv_abs = self.make_ci("absolute").predict_interval(*self.data(500, 4)[:1], self.data(500, 4)[2])
        self.assertLess(iv_abs.width_log.nunique(), 400)                           # clipping only; otherwise constant

    def test_bounds_and_units(self):
        ci = self.make_ci("normalized")
        X = pd.DataFrame({"x": np.linspace(0, 1, 50)})
        iv = ci.predict_interval(X, np.full(50, 0.05))           # forces lower clipping near 0
        self.assertTrue((iv.lo_ng_L >= -1e-9).all()); self.assertTrue((iv.hi_log >= iv.lo_log).all())
        self.assertTrue(iv.lower_clipped.any()); self.assertTrue((iv.lo_log >= 0).all())
        np.testing.assert_allclose(iv.pred_ng_L, 10 ** iv.pred_log - 1)
        self.assertTrue(iv.interval_valid.all())

    def test_clipping_never_reduces_coverage(self):
        ci = self.make_ci("normalized")
        Xt, yt, mt, _ = self.data(4000, 5)
        iv = ci.predict_interval(Xt, mt)
        raw_cov = ((yt >= iv.lo_log_raw) & (yt <= iv.hi_log)).mean(); clip_cov = ((yt >= iv.lo_log) & (yt <= iv.hi_log)).mean()
        self.assertAlmostEqual(raw_cov, clip_cov, places=12)    # y >= 0 always

    def test_unavailable_when_too_few_calibration_rows_or_nan(self):
        self.assertEqual(I.ConformalInterval._quantile(np.ones(5), 0.1), float("inf"))
        ci = self.make_ci("normalized")
        iv = ci.predict_interval(pd.DataFrame({"x": [0.2, 0.3]}), np.array([1.0, np.nan]))
        self.assertEqual(iv.interval_available.tolist(), [True, False]); self.assertFalse(iv.interval_valid.iloc[1])
        ci.q_ = float("inf")
        self.assertFalse(ci.predict_interval(pd.DataFrame({"x": [0.2]}), np.array([1.0])).interval_available.iloc[0])

    def test_coverage_table_groups_and_min_n(self):
        d = pd.DataFrame({"y_log": [1, 2, 3, 4], "lo_log": [0, 0, 0, 0], "hi_log": [1.5, 2.5, 2.5, 4.5], "width_log": 1.0,
                          "width_ng_L": 5.0, "interval_available": True, "g": ["a", "a", "b", "b"]})
        t = I.coverage_table(d, by="g", min_n=3)
        self.assertEqual(t.set_index("g").loc["a", "coverage"], 1.0); self.assertEqual(t.set_index("g").loc["b", "coverage"], 0.5)
        self.assertFalse(t.reliable_n.any())


class TestRisk(unittest.TestCase):
    cfg = R.RiskConfig(p_low=0.2, p_high=0.5, conc_moderate_ng_L=50, conc_high_ng_L=200, width_wide_log=1.0)

    def row(self, p, c, hi=None, w=0.5, ok=True, ood=False, ins=False, m="river"):
        hi = c * 1.5 if hi is None else hi
        return R.classify_row(p, c, hi, w, m, ok, ood, ins, self.cfg)

    def test_table_rows(self):
        self.assertEqual(self.row(0.05, 10)[0], "I")
        self.assertEqual(self.row(0.35, 20)[0], "II"); self.assertEqual(self.row(0.35, 100)[0], "II")
        self.assertEqual(self.row(0.8, 100)[0], "III"); self.assertEqual(self.row(0.8, 500)[0], "IV")
        self.assertEqual(self.row(0.8, 10)[0], "II")                       # unspecified combo, defined explicitly

    def test_uncertainty_categories(self):
        self.assertEqual(self.row(0.8, 500, w=0.2)[3], "High-confidence monitoring priority")
        self.assertIn("confirmatory", self.row(0.8, 500, w=2.0)[3])
        self.assertEqual(self.row(0.05, 10, w=0.2)[3], "Lower monitoring priority")
        self.assertIn("Insufficient evidence", self.row(0.05, 10, w=2.0)[3])
        self.assertIn("Intermediate", self.row(0.35, 10)[3])

    def test_uncertainty_never_means_low_risk(self):
        cls, *_, confirm = self.row(0.05, 10, w=2.0)
        self.assertEqual(cls, "II"); self.assertTrue(confirm)
        cfg = R.RiskConfig(**{**self.cfg.to_dict(), "uncertain_low_class": "I"})
        self.assertEqual(R.classify_row(0.05, 10, 15, 2.0, "river", True, False, False, cfg)[0], "I")
        self.assertEqual(self.row(0.8, 500, w=2.0)[0], "IV"); self.assertTrue(self.row(0.8, 500, w=2.0)[5])
        self.assertEqual(self.row(0.8, 10, hi=400, w=2.0)[0], "III")      # wide interval, upper bound high

    def test_missing_ood_unavailable(self):
        self.assertEqual(self.row(np.nan, 10)[0], "U"); self.assertEqual(self.row(0.8, 10, ins=True)[0], "U")
        self.assertEqual(self.row(0.05, 10, ood=True, w=0.1)[0], "II")     # OOD can never be Class I
        c = self.row(0.05, 10, ok=False, hi=np.nan, w=np.nan)
        self.assertEqual(c[0], "II"); self.assertTrue(c[5]); self.assertIn("unavailable", c[4][0])
        self.assertEqual(self.row(0.8, np.nan, ok=False, hi=np.nan, w=np.nan)[0], "III")

    def test_high_p_class_monotone_in_concentration(self):
        order = {"I": 1, "II": 2, "III": 3, "IV": 4}
        cl = [order[self.row(0.9, c, w=0.2)[0]] for c in (1, 10, 49, 51, 150, 199, 201, 1000)]
        self.assertEqual(cl, sorted(cl))

    def test_matrix_specific_and_benchmark(self):
        cfg = R.RiskConfig(**{**self.cfg.to_dict(), "conc_by_matrix": {"wastewater": (500, 2000)}})
        self.assertEqual(R.classify_row(0.9, 300, 400, 0.3, "wastewater", True, False, False, cfg)[0], "II")
        cfg.benchmarks = {"river": {"ng_L": 80, "source": "test", "jurisdiction": "test"}}
        self.assertEqual(R.classify_row(0.9, 100, 120, 0.3, "river", True, False, False, cfg)[0], "IV")

    def test_classify_uses_no_observed_columns_and_sensitivity(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"p_cal": rng.uniform(0, 1, 300), "pred_ng_L": rng.lognormal(4, 1.5, 300),
                           "hi_ng_L": rng.lognormal(5, 1.5, 300), "width_log": rng.uniform(0, 2, 300),
                           "water_matrix": "river", "interval_available": True, "flag_ood": False, "flag_insufficient_input": False})
        a = R.classify(df, self.cfg)
        df2 = df.assign(obs_occurrence=rng.integers(0, 2, 300), obs_conc_ng_L=rng.uniform(0, 1e4, 300))
        self.assertTrue(R.classify(df2, self.cfg).equals(a))
        s = R.sensitivity_analysis(df, self.cfg)
        np.testing.assert_allclose(s[[c for c in s if c.startswith("share_")]].sum(axis=1), 1.0)
        base = s[(s.p_low_x == 1.0) & (s.p_high == 0.5) & (s.conc_x == 1.0) & (s.width_x == 1.0)]
        self.assertEqual(len(base), 1); self.assertAlmostEqual(base.agreement_with_baseline.iloc[0], 1.0)

    def test_derive_thresholds_training_only(self):
        t = R.derive_thresholds(0.3, np.log10(np.arange(1, 201) + 1), np.array(["a"] * 100 + ["b"] * 100), np.linspace(0.2, 1.2, 50), min_matrix_n=30)
        self.assertEqual(set(t.conc_by_matrix), {"a", "b"}); self.assertLess(t.conc_moderate_ng_L, t.conc_high_ng_L)
        self.assertIn("NOT regulatory", t.provenance)


class TestOOD(unittest.TestCase):
    def test_flags(self):
        rng = np.random.default_rng(0)
        Xtr = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "m": rng.choice(["x", "y"], 500)})
        d = O.OODDetector(max_outside_frac=0.4).fit(Xtr)
        X = pd.DataFrame({"a": [0, 50, 0, np.nan], "b": [0, 50, 0, np.nan], "m": ["x", "x", "z", np.nan]})
        f = d.flags(X)
        self.assertEqual(f.flag_ood.tolist(), [False, True, True, False])
        self.assertEqual(f.flag_insufficient_input.tolist(), [False, False, False, True])


class TestExplainHelpers(unittest.TestCase):
    def test_names_and_aggregation_additive(self):
        names = ["num__rain", "cat__water_matrix_river", "cat__water_matrix_wastewater", "cat__season_wet"]
        self.assertEqual([E.original_name(n, ["water_matrix", "season"]) for n in names], ["rain", "water_matrix", "water_matrix", "season"])
        sv = np.random.default_rng(0).normal(size=(10, 4))
        agg = E.aggregate(sv, names, ["water_matrix", "season"])
        np.testing.assert_allclose(agg.sum(axis=1), sv.sum(axis=1)); self.assertEqual(list(agg.columns), ["rain", "water_matrix", "season"])

    def test_availability_reports_absence(self):
        a = E.feature_availability(["rainfall_7d", "pct_urban_land", "water_matrix", "season", "code_x"]).set_index("concept")
        self.assertTrue(a.loc["rainfall", "in_model_features"]); self.assertFalse(a.loc["pH", "in_model_features"])
        self.assertFalse(a.loc["COD", "in_model_features"])           # 'code_x' must not match COD

    def test_direction_and_top_contribs(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"u": rng.uniform(0, 1, 200), "m": rng.choice(["a", "b"], 200)})
        agg = pd.DataFrame({"u": 2 * X.u - 1, "m": np.where(X.m == "a", 0.5, -0.5)})
        self.assertIn("higher", E.direction_summary(agg, X, "u")["direction"])
        self.assertEqual(E.direction_summary(agg, X, "m")["highest_level"], "a")
        pos, neg = E.top_contribs(agg.head(3)); self.assertEqual(len(pos), 3)


def mkcfg(tmp, d):
    return P.RunConfig(out_dir=os.path.join(tmp, d), final_cls_candidates=("Random Forest", "CatBoost"),
                       final_reg_candidates=("Random Forest", "XGBoost"), min_pos_isotonic=40)


class TestPipelineEndToEnd(unittest.TestCase):
    """Runs the whole SMX pipeline on SYNTHETIC data (code-path test only)."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.A, cls.B = make_synth(0)
        cls.ctx = make_ctx(cls.A, cls.B)
        cls.res = P.run_smx_uq(cls.ctx, mkcfg(cls.tmp, "run1"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files_and_report(self):
        for f in self.res["files"]:
            self.assertTrue(os.path.exists(f), f)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "run1", "SMX_UQ_SHAP_RISK_REPORT.md")))

    def test_prediction_table_integrity(self):
        P_ = self.res["P"]
        self.assertEqual(len(P_), len(self.A)); self.assertTrue(P_.row_id.is_unique); self.assertTrue(P_.site_id.is_unique)
        num = P_.select_dtypes("number")
        self.assertFalse(np.isinf(num.values).any())
        self.assertFalse(P_["p_cal"].isna().any()); self.assertTrue(P_.p_cal.between(0, 1).all())
        ok = P_[P_.interval_available]
        self.assertTrue((ok.pred_upper90_ng_L >= ok.pred_lower90_ng_L).all()); self.assertTrue((ok.pred_lower90_ng_L >= -1e-9).all())
        self.assertTrue((ok.pred_conditional_conc_ng_L >= 0).all())
        np.testing.assert_allclose(ok.pred_conditional_conc_ng_L, 10 ** ok["pred_log10_Cplus1"] - 1, rtol=1e-9)
        self.assertTrue(set(P_.risk_class_code) <= {"I", "II", "III", "IV", "U"})
        self.assertTrue((P_.loc[P_.flag_ood, "risk_class_code"] != "I").all())

    def test_features_exclude_targets_and_split_is_country_disjoint(self):
        self.assertFalse(P.FORBIDDEN_AS_FEATURES & set(self.ctx["X_occ"].columns))
        Pm = self.res["P"]
        sets = {s: set(Pm.country[Pm.split == s]) for s in ("train", "val", "test")}
        self.assertFalse(sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])

    def test_guard_rejects_leaky_feature(self):
        ctx = make_ctx(self.A, self.B)
        ctx["X_occ"] = ctx["X_occ"].assign(LoQ=self.A.LoQ.values)
        with self.assertRaises(AssertionError):
            P.run_smx_uq(ctx, mkcfg(self.tmp, "leak"))

    def test_test_labels_never_influence_fitted_components(self):
        """Permute test-country labels (occurrence AND concentration): everything fitted/selected must be identical."""
        A2, B2 = self.A.copy(), self.B.copy()
        sp = make_ctx(self.A, self.B)["mask_A"]
        rng = np.random.default_rng(9)
        te_A = sp["test"]
        A2.loc[te_A, "Y_occ"] = rng.permutation(A2.loc[te_A, "Y_occ"].values)
        te_B = B2.country.isin(self.A.country[te_A].unique())
        B2.loc[te_B, "Y_conc"] = rng.permutation(B2.loc[te_B, "Y_conc"].values)
        ctx2 = make_ctx(A2, B2, split=None)
        res2 = P.run_smx_uq(ctx2, mkcfg(self.tmp, "run2"))
        r1 = self.res
        self.assertEqual(r1["cal"].params(), res2["cal"].params())
        self.assertEqual(r1["ci"].params(), res2["ci"].params())
        self.assertEqual(r1["thr"].to_dict(), res2["thr"].to_dict())
        a, b = r1["P"], res2["P"]
        pd.testing.assert_series_equal(a.risk_class_code, b.risk_class_code)
        np.testing.assert_allclose(a.p_cal, b.p_cal)
        self.assertNotEqual(r1["cov_overall"]["coverage"].tolist(), res2["cov_overall"]["coverage"].tolist())   # only evaluation changes

    def test_rerun_backs_up_instead_of_overwriting(self):
        P.run_smx_uq(make_ctx(self.A, self.B), mkcfg(self.tmp, "run1"))
        self.assertTrue([d for d in os.listdir(self.tmp) if d.startswith("run1_backup_")])

    def test_maps_png_written(self):
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "run1", "maps", "map_probability.png")))

    @unittest.skipUnless(E.HAS_SHAP, "shap not installed")
    def test_shap_outputs(self):
        self.assertTrue(self.res["shap_ran"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
