"""Orchestrates paper s3.13-3.15 for SMX ONLY. Reuses the notebook's fitted-by-recipe models (nothing is retuned).

Split discipline
  train countries : fit Stage-1/Stage-2 models, OOD detector, residual-scale model, concentration quantiles
  val countries   : probability calibrator, conformal quantile, risk-threshold derivation (labels or widths)
  test countries  : scored ONCE at the end; never fitted on, never used for any threshold or selection
"""
from __future__ import annotations
import json, os, shutil, time, platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, mean_absolute_error, precision_score, recall_score,
                             f1_score, confusion_matrix, r2_score)

from . import calibration as C, intervals as I, risk as R, ood as O, explain as E, maps as M

PCOLS = {"c_hat": "pred_conditional_conc_ng_L", "c_hi": "pred_upper90_ng_L", "width": "interval_width_log10"}
FORBIDDEN_AS_FEATURES = {"Y_occ", "Y_det", "Y_conc", "concentration_ng_L", "detection_status", "LoD", "LoQ"}


@dataclass
class RunConfig:
    out_dir: str = "smx_uq_outputs"
    alpha: float = 0.10
    interval_scale: str = "normalized"          # or 'absolute'
    min_pos_isotonic: int = 40
    min_group_n: int = 10                       # min quantified rows for per-group coverage to be flagged reliable
    final_cls_candidates: tuple = ("Random Forest", "XGBoost", "CatBoost")   # tree models -> exact TreeSHAP
    final_reg_candidates: tuple = ("Random Forest", "XGBoost", "CatBoost")
    q_mod: float = 0.50
    q_high: float = 0.80
    width_q: float = 0.50
    p_high: float = 0.50
    shap_max_rows: int = 2000
    min_rows_shap_reg: int = 100
    benchmarks: dict = field(default_factory=dict)   # {matrix: {"ng_L":..., "source":..., "jurisdiction":...}}
    make_maps: bool = True
    seed: int = 42


def prepare_outdir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        bak = p.with_name(p.name + "_backup_" + time.strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(p), str(bak))
    for sub in ("figures", "tables", "models", "maps"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def _f(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def md_table(df, index=False):
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def _reliability(ax, y, p, label, n_bins):
    try:
        fr, mp = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mp, fr, "o-", label=label)
    except ValueError:
        pass


def run_smx_uq(ctx: dict, cfg: RunConfig = RunConfig()):
    """ctx keys: df_occ, df_conc, X_occ, y_occ, X_conc, y_conc, cat_cols, mask_A, mask_B,
    candidates_cls {name: f(y_train)->unfitted pipeline}, candidates_reg {name: f()->unfitted pipeline},
    val_cls {name: val probs}, tau_cls {name: tau}, val_reg {name: val preds}, prep_factory()."""
    t0 = time.time()
    out = prepare_outdir(cfg.out_dir)
    FIG, TAB, MOD, MAP = out / "figures", out / "tables", out / "models", out / "maps"
    notes, files, res = [], [], {}

    A, B = ctx["df_occ"].reset_index(drop=True), ctx["df_conc"].reset_index(drop=True)
    XA, XB = ctx["X_occ"].reset_index(drop=True), ctx["X_conc"].reset_index(drop=True)
    yA, yB = np.asarray(ctx["y_occ"]).astype(int), np.asarray(ctx["y_conc"], float)
    cat_cols, features = list(ctx["cat_cols"]), list(XA.columns)
    mA, mB = ctx["mask_A"], ctx["mask_B"]
    assert not (FORBIDDEN_AS_FEATURES & set(features)), "target/detection-derived column present in features"
    assert len(XA) == len(A) and len(XB) == len(B)
    tr, va, te = mA["train"], mA["val"], mA["test"]
    trB, vaB, teB = mB["train"], mB["val"], mB["test"]
    assert not (set(A.country[tr]) & set(A.country[va])) and not (set(A.country[tr]) & set(A.country[te])) \
        and not (set(A.country[va]) & set(A.country[te])), "country overlap between splits"
    split = np.where(tr, "train", np.where(va, "val", "test"))

    def save(df, name, **kw):
        p = TAB / name
        df.to_csv(p, **kw); files.append(str(p)); return p

    def savefig(name):
        p = FIG / name; files.append(str(p)); return p

    # ------------------------------------------------------------------ model selection (validation only)
    cls_names = [n for n in ctx["candidates_cls"] if n in cfg.final_cls_candidates and n in ctx["val_cls"]]
    reg_names = [n for n in ctx["candidates_reg"] if n in cfg.final_reg_candidates and n in ctx["val_reg"]]
    sel_c = {n: average_precision_score(yA[va], ctx["val_cls"][n]) for n in cls_names}
    sel_r = {n: float(np.sqrt(np.mean((np.asarray(ctx["val_reg"][n]) - yB[vaB]) ** 2))) for n in reg_names}
    cls_name, reg_name = max(sel_c, key=sel_c.get), min(sel_r, key=sel_r.get)
    res["selected"] = {"classifier": cls_name, "regressor": reg_name, "val PR-AUC": sel_c, "val RMSE(log10)": sel_r}
    notes.append("Final models are chosen among single tree-based models by VALIDATION PR-AUC / RMSE so TreeSHAP is exact. "
                 "If a notebook ensemble scored higher, it is not used here.")

    # ------------------------------------------------------------------ Stage 1 refit + calibration
    pipe_c = ctx["candidates_cls"][cls_name](yA[tr]).fit(XA.loc[tr], yA[tr])
    raw = pipe_c.predict_proba(XA)[:, 1]
    chk = abs(average_precision_score(yA[va], raw[va]) - sel_c[cls_name])
    if chk > 1e-3:
        notes.append(f"Refit validation PR-AUC differs from the notebook registry by {chk:.4f} (non-deterministic fit?).")
    tau_raw = float(ctx["tau_cls"][cls_name])
    g_va = A["country"].values[va]
    method, sel_rows = C.select_method(raw[va], yA[va], g_va, cfg.min_pos_isotonic)
    cal = C.Calibrator(method).fit(raw[va], yA[va]) if method != "none" else C.Calibrator("none")
    p_cal = cal.predict(raw)
    p_cf = np.full(len(A), np.nan)
    if method == "none":
        p_cf[va] = raw[va]
    else:
        p_cf[va] = C.crossfit_calibrated(raw[va], yA[va], g_va, method)
    basis = np.where(va, np.where(np.isnan(p_cf), "in-sample (cross-fit unavailable)", "cross-fitted"),
                     np.where(te, "held-out", "training-country (model fitted on these rows)"))
    p_used = np.where(va & ~np.isnan(p_cf), p_cf, p_cal)
    res["calibration_method"] = method
    save(pd.DataFrame(sel_rows), "calibration_method_selection_validation_crossfit.csv", index=False)

    rows = []
    for nm, idx, pr_raw, pr_cal in (("validation (cross-fitted calibrator)", va, raw[va], p_cf[va]),
                                    ("test (held-out countries)", te, raw[te], p_cal[te])):
        yy = yA[idx]
        ok = ~np.isnan(pr_cal)
        if len(np.unique(yy[ok])) < 2:
            notes.append(f"Calibration evaluation on {nm}: single class, metrics undefined."); continue
        for lab, pp in (("uncalibrated", pr_raw[ok]), (f"calibrated ({method})", pr_cal[ok])):
            rows.append({"set": nm, "probabilities": lab, **C.calibration_report(yy[ok], pp)})
    cal_tab = pd.DataFrame(rows)
    save(cal_tab, "calibration_comparison.csv", index=False)
    res["calibration_table"] = cal_tab

    # reliability diagram
    nb = int(max(3, min(10, yA[te].sum() // 5))) if yA[te].sum() > 0 else 5
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (nm, idx, pr, pc) in zip(axs, (("Validation (cross-fitted)", va, raw[va], p_cf[va]), ("Test (held-out)", te, raw[te], p_cal[te]))):
        ok = ~np.isnan(pc)
        if ok.sum() and len(np.unique(yA[idx][ok])) == 2:
            _reliability(ax, yA[idx][ok], pr[ok], "uncalibrated", nb)
            _reliability(ax, yA[idx][ok], pc[ok], f"calibrated ({method})", nb)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction")
        ax.set_title(f"{nm}: n={int(ok.sum())}, positives={int(yA[idx][ok].sum())}", fontsize=9); ax.legend(fontsize=8)
    plt.suptitle("SMX occurrence reliability (quantile bins; few positives -> noisy)", fontsize=10)
    plt.tight_layout(); plt.savefig(savefig("calibration_reliability.png"), dpi=200); plt.close()

    # per-country calibration in the large (test countries)
    pc_rows = [{"country": c, "n": int(m.sum()), "prevalence": float(yA[te][m].mean()),
                "mean_uncalibrated": float(raw[te][m].mean()), "mean_calibrated": float(p_cal[te][m].mean())}
               for c in pd.unique(A["country"].values[te]) for m in [A["country"].values[te] == c]]
    save(pd.DataFrame(pc_rows), "calibration_by_test_country.csv", index=False)

    pred_te = (raw[te] >= tau_raw).astype(int)
    tn, fp, fn, tp = confusion_matrix(yA[te], pred_te, labels=[0, 1]).ravel()
    op = {"tau_raw (from notebook, validation-selected)": tau_raw, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
          "Precision": precision_score(yA[te], pred_te, zero_division=0), "Recall": recall_score(yA[te], pred_te, zero_division=0),
          "F1": f1_score(yA[te], pred_te, zero_division=0)}
    save(pd.Series(op).to_frame("test").reset_index(names="metric"), "stage1_operating_point_test.csv", index=False)
    res["operating_point"] = op
    res["class_balance"] = {s: {"n": int(m.sum()), "positives": int(yA[m].sum()), "prevalence": float(yA[m].mean())}
                            for s, m in (("train", tr), ("val", va), ("test", te))}

    # ------------------------------------------------------------------ Stage 2 + conformal intervals
    mk_reg = ctx["candidates_reg"][reg_name]
    pipe_r = mk_reg().fit(XB.loc[trB], yB[trB])
    yhat_A = np.clip(pipe_r.predict(XA), 0, None)
    yhat_B = np.clip(pipe_r.predict(XB), 0, None)
    oof = I.grouped_oof_regression(mk_reg, XB.loc[trB], yB[trB], B["country"].values[trB])
    resid = np.abs(yB[trB] - oof)
    gB = B["country"].values
    ci = I.ConformalInterval(cfg.alpha, cfg.interval_scale, seed=cfg.seed).fit_scale(ctx["prep_factory"], XB.loc[trB], resid)
    ci.calibrate(XB.loc[vaB], yB[vaB], yhat_B[vaB])
    if not np.isfinite(ci.q_):
        notes.append(f"Conformal quantile unavailable: only {ci.n_cal_} calibration rows (need >= {int(np.ceil(1/cfg.alpha - 1))}). "
                     "All intervals flagged unavailable.")
    ci_abs = I.ConformalInterval(cfg.alpha, "absolute").fit_scale(ctx["prep_factory"], XB.loc[trB], resid)
    ci_abs.calibrate(XB.loc[vaB], yB[vaB], yhat_B[vaB])
    ivA = ci.predict_interval(XA, yhat_A)
    ivB = ci.predict_interval(XB, yhat_B)
    ivB_abs = ci_abs.predict_interval(XB, yhat_B)

    evB = ivB.copy()
    evB["y_log"] = yB; evB["country"] = gB
    evB["water_matrix"] = B["water_matrix"].values if "water_matrix" in B else "NA"
    evB_abs = ivB_abs.copy(); evB_abs["y_log"] = yB
    cmp_rows = []
    for nm, d in (("normalized conformal", evB[teB]), ("absolute-residual conformal", evB_abs[teB])):
        r = I.coverage_table(d).iloc[0].to_dict(); r["method"] = nm; cmp_rows.append(r)
    cov_overall = pd.DataFrame(cmp_rows)[["method", "n", "n_with_interval", "coverage", "mean_width_log10", "mean_width_ng_L", "median_width_ng_L"]]
    cov_country = I.coverage_table(evB[teB], by="country", min_n=cfg.min_group_n)
    cov_matrix = I.coverage_table(evB[teB], by="water_matrix", min_n=cfg.min_group_n)
    loo = I.leave_one_calibration_group_out(ci.cal_scores_, gB[vaB], cfg.alpha)
    for nm, d in (("interval_coverage_test_overall.csv", cov_overall), ("interval_coverage_test_by_country.csv", cov_country),
                  ("interval_coverage_test_by_matrix.csv", cov_matrix), ("interval_leave_one_calibration_country_out.csv", loo)):
        save(d, nm, index=False)
    from scipy.stats import spearmanr
    tb = evB[teB]
    sp = spearmanr(tb["sigma_log"], np.abs(tb["y_log"] - tb["pred_log"])).statistic if len(tb) > 5 else np.nan
    s2 = {"RMSE_log10": float(np.sqrt(np.mean((yhat_B[teB] - yB[teB]) ** 2))), "MAE_log10": float(mean_absolute_error(yB[teB], yhat_B[teB])),
          "R2": float(r2_score(yB[teB], yhat_B[teB])) if teB.sum() > 1 else np.nan,
          "MAE_ng_L": float(mean_absolute_error(I.to_conc(yB[teB]), I.to_conc(yhat_B[teB]))),
          "spearman(sigma_hat, |error|) test": float(sp), "conformal_q": ci.q_, "n_cal": ci.n_cal_}
    res.update(cov_overall=cov_overall, cov_country=cov_country, cov_matrix=cov_matrix, loo=loo, stage2_test=s2)
    save(pd.Series(s2).to_frame("value").reset_index(names="metric"), "stage2_point_metrics_test.csv", index=False)

    # ------------------------------------------------------------------ OOD + thresholds + risk
    det = O.OODDetector().fit(XA.loc[tr])
    fl = det.flags(XA)
    thr = R.derive_thresholds(float(yA[tr].mean()), yB[trB], (B["water_matrix"].values[trB] if "water_matrix" in B else np.array(["NA"] * trB.sum())),
                              ivB["width_log"].values[vaB], cfg.q_mod, cfg.q_high, cfg.width_q, p_high=cfg.p_high)
    thr.benchmarks = dict(cfg.benchmarks)

    P = pd.DataFrame({"row_id": np.arange(len(A))})
    P["site_id"] = A["site_id"].values if "site_id" in A else np.arange(len(A))
    for c in ("country", "water_matrix", "latitude", "longitude", "year"):
        if c in A: P[c] = A[c].values
    P["split"] = split
    P["held_out"] = ~tr
    P["raw_occurrence_score"] = raw
    P["p_calibrated"] = p_cal
    P["p_calibrated_crossfit_val"] = p_cf
    P["p_cal"] = p_used
    P["p_cal_basis"] = basis
    P["occ_predicted_at_tau"] = (raw >= tau_raw).astype(int)
    P = pd.concat([P, ivA.reset_index(drop=True).rename(columns={"pred_log": "pred_log10_Cplus1"}).rename(
        columns={"pred_ng_L": "pred_ng_L"}), fl.reset_index(drop=True)], axis=1)
    P["conc_prediction_scope"] = "conditional on quantification (not unconditional expected concentration)"
    rk = R.classify(P, thr)
    P = pd.concat([P, rk.reset_index(drop=True)], axis=1)
    P = P.rename(columns={"lo_ng_L": "pred_lower90_ng_L", "hi_ng_L": "pred_upper90_ng_L", "width_ng_L": "interval_width_ng_L",
                          "width_log": "interval_width_log10", "pred_ng_L": "pred_conditional_conc_ng_L"})
    P["obs_occurrence"] = yA
    if "concentration_ng_L" in A: P["obs_conc_ng_L"] = A["concentration_ng_L"].values
    if "detection_status" in A: P["obs_detection_status"] = A["detection_status"].values
    save(P, "smx_site_predictions_all.csv", index=False)
    eval_cols = [c for c in P.columns if c != "conc_prediction_scope"]
    save(P[P.held_out][eval_cols], "smx_site_predictions_heldout.csv", index=False)
    save(P[["row_id", "site_id", "split", "p_cal_basis", "raw_occurrence_score", "p_calibrated", "p_calibrated_crossfit_val"]],
         "smx_calibrated_occurrence_probabilities.csv", index=False)
    tcols = ["site_id", "country", "latitude", "longitude", "water_matrix", "split", "p_calibrated", "pred_conditional_conc_ng_L",
             "pred_lower90_ng_L", "pred_upper90_ng_L", "interval_width_ng_L", "interval_available", "interval_valid", "lower_clipped",
             "flag_ood", "uncertainty_category", "risk_class", "monitoring_recommendation"]
    table_B = P[[c for c in tcols if c in P]]
    save(table_B, "smx_concentration_prediction_table.csv", index=False)
    save(P[P.held_out].sort_values("risk_class_code", ascending=False)[
        [c for c in ["site_id", "country", "water_matrix", "split", "p_cal", "pred_conditional_conc_ng_L", "pred_lower90_ng_L",
                     "pred_upper90_ng_L", "uncertainty_category", "risk_class", "monitoring_recommendation", "reasons"] if c in P]],
         "smx_risk_priority_table_heldout.csv", index=False)
    with open(MOD / "risk_thresholds.json", "w") as fh:
        json.dump(thr.to_dict(), fh, indent=2, default=float)
    files.append(str(MOD / "risk_thresholds.json"))
    sens = R.sensitivity_analysis(P[P.held_out], thr, cols=PCOLS)
    save(sens, "risk_threshold_sensitivity.csv", index=False)
    ho = P[P.held_out]
    dist = pd.crosstab(ho["split"], ho["risk_class_code"], margins=True)
    save(dist.reset_index(), "risk_class_distribution_heldout.csv", index=False)
    unc = pd.crosstab(ho["split"], ho["uncertainty_category"])
    save(unc.reset_index(), "uncertainty_category_distribution_heldout.csv", index=False)
    pt = P[P.split == "test"]
    ev_rows = []
    for k, g in pt.groupby("risk_class_code"):
        r = {"risk_class_code": k, "n": len(g), "observed_occurrence_rate": float(g["obs_occurrence"].mean())}
        if "obs_conc_ng_L" in g and (g["obs_occurrence"] == 1).any():
            r["median_observed_conc_ng_L_quantified"] = float(g.loc[g["obs_occurrence"] == 1, "obs_conc_ng_L"].median())
        ev_rows.append(r)
    evalc = pd.DataFrame(ev_rows)
    save(evalc, "risk_class_posthoc_evaluation_test.csv", index=False)
    res.update(thr=thr, risk_dist=dist, unc_dist=unc, sens=sens, evalc=evalc)

    # ------------------------------------------------------------------ SHAP
    shap_ok = E.HAS_SHAP
    avail = E.feature_availability(features, XA)
    save(avail, "shap_requested_feature_availability.csv", index=False)
    res["feature_availability"] = avail
    res["shap_ran"] = False
    if not shap_ok:
        notes.append("shap is not installed: SHAP tables/figures NOT generated (install `shap`, rerun).")
    else:
        try:
            _run_shap(ctx, cfg, res, notes, save, savefig, pipe_c, pipe_r, XA, XB, yA, yB, A, B, P, cat_cols, features, va, te, vaB, teB, tau_raw)
            res["shap_ran"] = True
        except Exception as e:     # keep the rest of the outputs
            import traceback
            notes.append(f"SHAP step failed: {e!r}")
            res["shap_error"] = traceback.format_exc()

    # ------------------------------------------------------------------ maps
    res["maps"] = {}
    if cfg.make_maps:
        mP = P.copy()
        mP["uncertainty_code"] = mP["uncertainty_category"]
        ucol = {"High-confidence monitoring priority": "#1a9850", "High priority but uncertain; confirmatory monitoring required": "#d73027",
                "Lower monitoring priority": "#91bfdb", "Insufficient evidence; additional data required": "#fdae61",
                "Intermediate probability; narrow interval": "#ffffbf", "Intermediate probability; wide interval, confirmatory data advised": "#fee090",
                "Unavailable inputs": "#7f7f7f"}
        jobs = [("p_cal", "SMX calibrated occurrence probability", "map_probability", False, "plasma", None, "Calibrated P(SMX quantified)"),
                ("interval_width_log10", "SMX 90% interval width (log10(C+1) units; conditional concentration)", "map_uncertainty_width", False, "magma_r", None, "width (log10 units)"),
                ("uncertainty_code", "SMX uncertainty categories", "map_uncertainty_category", True, None, ucol, "category"),
                ("risk_class_code", "SMX monitoring risk classes", "map_risk_class", True, None, None, "class")]
        nvalid = int(M.valid_coords(mP).sum())
        res["maps"]["sites_with_valid_coordinates"] = nvalid
        if nvalid == 0:
            notes.append("No valid coordinates: maps not drawn; underlying data are in the prediction CSVs.")
        for col, ttl, nm, cat, cmap, colors, lab in jobs:
            if nvalid == 0: break
            n = M.static_map(mP, col, ttl, str(MAP / f"{nm}.png"), cmap=cmap or "viridis", categorical=cat, label=lab, colors=colors)
            files.append(str(MAP / f"{nm}.png"))
            ok = M.html_map(mP, col, ttl, str(MAP / f"{nm}.html"),
                            ["site_id", "country", "water_matrix", "split", "p_cal", "pred_conditional_conc_ng_L", "pred_lower90_ng_L",
                             "pred_upper90_ng_L", "uncertainty_category", "risk_class"], categorical=cat, colors=colors)
            if ok: files.append(str(MAP / f"{nm}.html"))
            res["maps"][nm] = {"sites_plotted": n, "html": ok}
        if not M.HAS_FOLIUM:
            notes.append("folium not installed: interactive HTML maps NOT generated (static PNG maps and CSVs were).")

    # ------------------------------------------------------------------ artifacts + metadata
    joblib.dump({"classifier_pipeline": pipe_c, "regressor_pipeline": pipe_r, "calibrator": cal, "conformal": ci,
                 "ood_detector": det, "risk_config": thr}, MOD / "smx_uq_models.joblib")
    files.append(str(MOD / "smx_uq_models.joblib"))
    meta = {"selected": res["selected"], "calibration": cal.params(), "conformal": ci.params(), "tau_raw": tau_raw,
            "split_countries": {s: sorted(map(str, pd.unique(A["country"].values[m]))) for s, m in (("train", tr), ("val", va), ("test", te))},
            "features": features, "seed": cfg.seed, "config": {k: v for k, v in asdict(cfg).items()},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "runtime_s": round(time.time() - t0, 1), "notes": notes}
    with open(MOD / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    files.append(str(MOD / "metadata.json"))
    res.update(P=P, notes=notes, files=files, cal=cal, ci=ci)
    _write_report(res, cfg, out, meta)
    files.append(str(out / "SMX_UQ_SHAP_RISK_REPORT.md"))
    return res


# ======================================================================================================
def _run_shap(ctx, cfg, res, notes, save, savefig, pipe_c, pipe_r, XA, XB, yA, yB, A, B, P, cat_cols, features, va, te, vaB, teB, tau_raw):
    held = va | te
    idx = np.where(held)[0]
    if len(idx) > cfg.shap_max_rows:
        idx = np.random.default_rng(cfg.seed).choice(idx, cfg.shap_max_rows, replace=False)
    Xh = XA.iloc[idx]
    sc = E.compute_shap(pipe_c, Xh, "cls")
    agg_c = E.aggregate(sc["values"], sc["names"], cat_cols, index=idx)
    imp_c = E.importance_table(agg_c)
    save(imp_c, "shap_importance_classifier.csv", index=False)
    E.beeswarm(agg_c, Xh, f"SMX occurrence classifier: SHAP (held-out countries)\noutput = {sc['output']}", savefig("shap_beeswarm_classifier.png"))
    E.bar_plot(imp_c, f"Classifier mean |SHAP|\noutput = {sc['output']}", savefig("shap_importance_bar_classifier.png"))
    res["shap_cls"] = {"output": sc["output"], "n_rows": len(idx), "importance": imp_c}

    # regressor global (observed quantified held-out rows) and for site table (all held-out A rows)
    reg_ok = int((vaB | teB).sum()) >= cfg.min_rows_shap_reg
    XBh = XB.loc[vaB | teB]
    if reg_ok:
        sr = E.compute_shap(pipe_r, XBh, "reg")
        agg_r = E.aggregate(sr["values"], sr["names"], cat_cols, index=XBh.index)
        imp_r = E.importance_table(agg_r)
        save(imp_r, "shap_importance_regressor.csv", index=False)
        E.beeswarm(agg_r, XBh, f"SMX concentration regressor: SHAP (held-out quantified samples)\noutput = {sr['output']}", savefig("shap_beeswarm_regressor.png"))
        E.bar_plot(imp_r, f"Regressor mean |SHAP|\noutput = {sr['output']}", savefig("shap_importance_bar_regressor.png"))
        res["shap_reg"] = {"output": sr["output"], "n_rows": len(XBh), "importance": imp_r}
    else:
        notes.append(f"Regressor SHAP skipped: only {int((vaB | teB).sum())} held-out quantified rows (< {cfg.min_rows_shap_reg}).")
    sr_site = E.compute_shap(pipe_r, Xh, "reg")
    agg_rs = E.aggregate(sr_site["values"], sr_site["names"], cat_cols, index=idx)

    # direction of effect + dependence plots
    dirs, dep_files = [], []
    imp_rank = imp_c["feature"].tolist()
    for concept in E.CONCEPTS:
        for f in E.match(features, concept):
            if f not in agg_c: continue
            d = E.direction_summary(agg_c, Xh, f); d["concept"] = concept; d["model"] = "classifier"; dirs.append(d)
            partner = next((o for o in imp_rank if o != f), None)
            fn = f"shap_dependence_classifier_{f}.png".replace("/", "_")
            E.dependence(agg_c, Xh, f, partner, savefig(fn), f"{f}: association with classifier score (colour: {partner})")
            dep_files.append(fn)
    if reg_ok:
        for f in imp_r["feature"].head(6):
            d = E.direction_summary(agg_r, XBh, f); d["concept"] = "top-6 regressor"; d["model"] = "regressor"; dirs.append(d)
            fn = f"shap_dependence_regressor_{f}.png".replace("/", "_")
            E.dependence(agg_r, XBh, f, None, savefig(fn), f"{f}: association with log10(C+1)")
    for f in imp_c["feature"].head(6):           # always give the model's own top features too
        if f in agg_c and f"shap_dependence_classifier_{f}.png" not in dep_files:
            E.dependence(agg_c, Xh, f, next((o for o in imp_rank if o != f), None), savefig(f"shap_dependence_classifier_{f}.png"),
                         f"{f}: association with classifier score")
            d = E.direction_summary(agg_c, Xh, f); d["concept"] = "top-6 classifier"; d["model"] = "classifier"; dirs.append(d)
    save(pd.DataFrame(dirs), "shap_direction_of_effect.csv", index=False)
    res["directions"] = pd.DataFrame(dirs)

    # interactions
    mat, payload = E.interaction_strengths(pipe_c.named_steps["model"], sc["Z"], sc["names"], cat_cols, seed=cfg.seed)
    irows = []
    if mat is None:
        notes.append("SHAP interaction values unavailable: " + str(payload) + ". Dependence plots coloured by the partner feature are the fallback.")
        res["interaction_note"] = str(payload)
    else:
        save(mat.reset_index(names="feature"), "shap_interaction_strength_classifier.csv", index=False)
        ids, T, feats = payload
        for a_c, b_c in E.INTERACTIONS:
            fa, fb = E.match(features, a_c), E.match(features, b_c)
            if not fa or not fb:
                irows.append({"pair": f"{a_c} x {b_c}", "status": "not testable: " + ("; ".join(n for n, l in ((a_c, fa), (b_c, fb)) if not l) + " absent from the model features")}); continue
            for xa in fa:
                for xb in fb:
                    if xa in feats and xb in feats and xa != xb:
                        ia, ib = feats.index(xa), feats.index(xb)
                        v = 2 * T[:, ia, ib]
                        irows.append({"pair": f"{xa} x {xb}", "status": "computed", "mean_abs_interaction": float(np.abs(v).mean()),
                                      "rank_among_all_pairs": int((mat.values[np.triu_indices(len(feats), 1)] > np.abs(v).mean()).sum() + 1)})
                        fn = f"shap_interaction_{xa}_x_{xb}.png".replace("/", "_")
                        plt.figure(figsize=(6, 4.2))
                        xr = Xh.iloc[ids][xa]
                        if pd.api.types.is_numeric_dtype(xr):
                            m_ = xr.notna().values
                            col = Xh.iloc[ids][xb]
                            cvals = col.values[m_] if pd.api.types.is_numeric_dtype(col) else pd.Categorical(col.values[m_]).codes
                            plt.scatter(xr.values[m_], v[m_], c=cvals, cmap="viridis", s=14)
                            plt.colorbar(label=xb + ("" if pd.api.types.is_numeric_dtype(col) else " (category code)"))
                        else:
                            plt.scatter(pd.Categorical(xr.values).codes, v, s=14)
                            plt.xticks(range(len(pd.Categorical(xr.values).categories)), pd.Categorical(xr.values).categories, rotation=30, fontsize=7)
                        plt.axhline(0, color="k", lw=0.6); plt.ylabel("SHAP interaction value (log-odds units)")
                        plt.xlabel(xa); plt.title(f"Interaction {xa} x {xb} (association, not causation)", fontsize=9)
                        plt.tight_layout(); plt.savefig(savefig(fn), dpi=200); plt.close()
        top = mat.where(np.triu(np.ones(mat.shape, bool), 1)).stack().sort_values(ascending=False).head(10)
        res["top_interactions"] = top.reset_index().rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "mean_abs_interaction"})
        save(res["top_interactions"], "shap_top_interaction_pairs.csv", index=False)
    save(pd.DataFrame(irows), "shap_requested_interactions.csv", index=False)
    res["requested_interactions"] = pd.DataFrame(irows)

    # site-level + representative local explanations (held-out only)
    pos_c, neg_c = E.top_contribs(agg_c)
    pos_r, neg_r = E.top_contribs(agg_rs)
    S = P.iloc[idx].copy()
    S["occ_top_positive_shap"], S["occ_top_negative_shap"] = pos_c.values, neg_c.values
    S["conc_top_positive_shap"], S["conc_top_negative_shap"] = pos_r.values, neg_r.values
    S["occ_shap_output"] = sc["output"]
    S["occ_error_type"] = np.where((S.occ_predicted_at_tau == 1) & (S.obs_occurrence == 0), "false_positive (classifier)",
                           np.where((S.occ_predicted_at_tau == 0) & (S.obs_occurrence == 1), "false_negative (classifier)", "correct"))
    if "obs_conc_ng_L" in S:
        S["conc_abs_error_ng_L_if_quantified"] = np.where(S.obs_occurrence == 1, (S.pred_conditional_conc_ng_L - S.obs_conc_ng_L).abs(), np.nan)
    keep = [c for c in ["row_id", "site_id", "country", "water_matrix", "split", "p_cal", "pred_conditional_conc_ng_L", "pred_lower90_ng_L",
                        "pred_upper90_ng_L", "uncertainty_category", "risk_class", "obs_occurrence", "obs_conc_ng_L", "occ_error_type",
                        "conc_abs_error_ng_L_if_quantified", "occ_top_positive_shap", "occ_top_negative_shap",
                        "conc_top_positive_shap", "conc_top_negative_shap"] if c in S]
    save(S[keep], "smx_site_explanations_heldout.csv", index=False)

    T_ = S[S.split == "test"] if (S.split == "test").any() else S
    cases = {}
    cases["highest_occurrence_probability"] = T_["p_cal"].idxmax()
    cases["highest_predicted_concentration"] = T_["pred_conditional_conc_ng_L"].idxmax()
    fp = T_[T_.occ_error_type.str.startswith("false_positive")]
    fn_ = T_[T_.occ_error_type.str.startswith("false_negative")]
    if len(fp): cases["false_positive_highest_p"] = fp["p_cal"].idxmax()
    if len(fn_): cases["false_negative_lowest_p"] = fn_["p_cal"].idxmin()
    av = T_[T_.interval_available]
    if len(av): cases["highest_uncertainty_widest_interval"] = av["interval_width_log10"].idxmax()
    pos_in = {int(i): k for k, i in enumerate(idx)}
    crow = []
    for nm, ridx in cases.items():
        k = pos_in[int(ridx)]
        r = S.loc[ridx]
        contrib = agg_c.iloc[k]
        E.local_plot(contrib, sc["base_value"], f"p_cal={r.p_cal:.2f}, raw score contribution sum + base {sc['base_value']:+.2f}; observed={int(r.obs_occurrence)}",
                     f"[{nm}] occurrence classifier: {r.site_id} ({r.country})", savefig(f"shap_local_classifier_{nm}.png"))
        E.local_plot(agg_rs.iloc[k], sr_site["base_value"], f"conditional prediction = {r.pred_conditional_conc_ng_L:.1f} ng/L "
                     f"[{r.pred_lower90_ng_L:.1f}, {r.pred_upper90_ng_L:.1f}]",
                     f"[{nm}] concentration regressor (conditional): {r.site_id}", savefig(f"shap_local_regressor_{nm}.png"))
        crow.append({"case": nm, **{c: r[c] for c in keep if c in r.index}})
    save(pd.DataFrame(crow), "smx_representative_cases.csv", index=False)
    res["cases"] = pd.DataFrame(crow)


# ======================================================================================================
def _write_report(res, cfg, out, meta):
    P, thr = res["P"], res["thr"]
    L = []
    w = L.append
    w("# SMX: uncertainty, SHAP explainability and monitoring risk classes\n")
    w("> Auto-generated by `smx_uq.run_smx_uq` from the actual run. Every number below was computed in this run; interpretive text "
      "is limited to what the tables show. SMX only; CIP untouched. SHAP and risk classes describe **associations learned from "
      "observational monitoring data**, not causes.\n")
    w("## 1. Existing SMX pipeline and what was added\n")
    w(f"* Two-stage hurdle pipeline from the notebook (Stage 1 occurrence `Y_occ`; Stage 2 `log10(C+1)` for quantified samples). "
      f"Features exclude `country`, coordinates, IDs, LoD/LoQ, detection status and concentration; the run asserts this.\n"
      f"* Country-grouped split reused unchanged: train {meta['split_countries']['train'].__len__()} / val {meta['split_countries']['val'].__len__()} / "
      f"test {meta['split_countries']['test'].__len__()} countries (list in `models/metadata.json`).\n"
      f"* Final models (single tree models, chosen on **validation** PR-AUC / RMSE only): classifier **{res['selected']['classifier']}**, "
      f"regressor **{res['selected']['regressor']}**. Hyperparameters are the notebook's tuned values (not re-tuned); both were refit on training countries.\n"
      f"* Added: calibration, conformal intervals, OOD flags, uncertainty categories, risk classes, SHAP, maps.\n")
    cb = pd.DataFrame(res["class_balance"]).T.astype({"n": int, "positives": int})
    w("Class balance:\n"); w(md_table(cb, index=True) + "\n")
    w("## 2. Probability calibration (paper 3.13.1)\n")
    w(f"Method chosen by country-grouped cross-fitted log loss on the validation countries: **{res['calibration_method']}**. "
      f"Isotonic needs >= {cfg.min_pos_isotonic} validation positives; the calibrator never sees test labels.\n")
    w(md_table(pd.read_csv(Path(cfg.out_dir) / "tables" / "calibration_method_selection_validation_crossfit.csv")) + "\n")
    w("Raw vs calibrated (validation rows use a cross-fitted calibrator; test rows are held-out):\n")
    w(md_table(res["calibration_table"]) + "\n")
    ct = res["calibration_table"]
    if len(ct):
        t = ct[ct["set"].str.startswith("test")]
        if len(t) == 2:
            b0, b1 = t.iloc[0], t.iloc[1]
            if res["calibration_method"] == "none":
                w("**Reading:** no calibrator was adopted: on the validation countries (cross-fitted) neither Platt scaling nor isotonic regression "
                  "beat the raw scores on log loss, so the 'calibrated' probabilities equal the raw ones. Test-country reliability "
                  f"of the raw scores: Brier {b0['Brier']:.3f}, log loss {b0['LogLoss']:.3f}, ECE {b0['ECE']:.3f}, calibration slope {b0['Cal. slope']:.2f} "
                  f"({int(b0['n'])} rows, {int(b0['n_pos'])} positives).\n")
            else:
                better = (b1["Brier"] < b0["Brier"]) and (b1["LogLoss"] < b0["LogLoss"])
                w(f"**Reading:** on the test countries Brier {b0['Brier']:.3f} -> {b1['Brier']:.3f}, log loss {b0['LogLoss']:.3f} -> {b1['LogLoss']:.3f}, "
                  f"ECE {b0['ECE']:.3f} -> {b1['ECE']:.3f}. Calibration {'improved' if better else 'did NOT clearly improve'} probability reliability on held-out countries "
                  f"(single test sample of {int(b0['n'])} rows, {int(b0['n_pos'])} positives: treat differences as indicative).\n")
    w("Limitations: the base classifier uses class-weighted losses, which inflates raw probabilities, so raw scores are not probabilities; "
      "calibration fitted on a few validation countries assumes the score-to-outcome relation transfers to new countries (prevalence shift is the main risk); "
      "few positives make reliability bins noisy. Figure: `figures/calibration_reliability.png`. "
      "Confusion matrix at the notebook's validation-selected threshold (decisions unchanged by monotone calibration):\n")
    w(md_table(pd.Series(res["operating_point"]).to_frame("test").reset_index(names="metric")) + "\n")
    w("## 3. Concentration prediction intervals (paper 3.13.2)\n")
    w("Method: normalised split-conformal on `y = log10(C+1)`. Score `|y - yhat| / sigma(x)`; sigma(x) from a random forest fitted on country-grouped "
      "out-of-fold absolute residuals of the **training** countries; calibration quantile from the **validation** countries' quantified samples "
      f"(n = {res['ci'].n_cal_}, q = {res['ci'].q_:.3f}). Bounds are back-transformed with `C = 10^y - 1`. "
      "The lower bound is intersected with the support (`y >= 0`, i.e. `C >= 0`); this cannot reduce coverage and every clipped row is flagged (`lower_clipped`). "
      "Coverage guarantees need exchangeability between calibration and new samples; samples within a country are dependent and test countries are new, "
      "so coverage is only approximate and is reported per country. Intervals are **conditional on the sample being quantified**; they say nothing about occurrence "
      "and are not an unconditional expected concentration. Quantified-only evaluation: non-quantified samples have no observed concentration, so coverage cannot be assessed on them.\n")
    w("Test-country coverage (nominal 90%):\n"); w(md_table(res["cov_overall"]) + "\n")
    w("By country:\n"); w(md_table(res["cov_country"]) + "\n")
    w("By water matrix:\n"); w(md_table(res["cov_matrix"]) + "\n")
    w("Country-level spread of the calibration step (leave-one-calibration-country-out):\n"); w(md_table(res["loo"]) + "\n")
    w("Stage-2 point metrics on test (preserved scale: log10, plus ng/L):\n"); w(md_table(pd.Series(res["stage2_test"]).to_frame("value").reset_index(names="metric")) + "\n")
    w("## 4. SHAP (paper 3.14)\n")
    if not res.get("shap_ran"):
        w("**SHAP was not generated in this run** (see limitations below).\n")
    else:
        w(f"Explained outputs: classifier = *{res['shap_cls']['output']}* (this is the **uncalibrated** score; calibration is monotone, so direction of effect is unchanged but magnitudes are not probability units); "
          f"regressor = *{res.get('shap_reg', {}).get('output', 'n/a')}*. Explained rows: held-out (validation + test) countries only. TreeSHAP; one-hot columns are summed back to the original feature.\n")
        w("Global ranking (classifier, top 15):\n"); w(md_table(res["shap_cls"]["importance"].head(15)) + "\n")
        if "shap_reg" in res:
            w("Global ranking (regressor, top 15):\n"); w(md_table(res["shap_reg"]["importance"].head(15)) + "\n")
        w("Requested features, availability in the model (absence is a finding, not a gap to fill):\n"); w(md_table(res["feature_availability"]) + "\n")
        d = res["directions"]
        if len(d):
            w("Direction of effect (data-driven; Spearman of raw value vs SHAP for numeric features, mean SHAP per level for categorical):\n")
            w(md_table(d[[c for c in ["model", "concept", "feature", "type", "n", "spearman_value_vs_shap", "direction"] if c in d]]) + "\n")
        w("Interactions requested:\n"); w(md_table(res["requested_interactions"]) + "\n")
        if "top_interactions" in res:
            w("Strongest interaction pairs overall (mean |SHAP interaction| x2):\n"); w(md_table(res["top_interactions"]) + "\n")
        elif "interaction_note" in res:
            w(f"Interaction values not computed: {res['interaction_note']}\n")
        if "cases" in res:
            w("Representative held-out cases (selected after the fact for illustration; test labels are used only to pick false positives/negatives, never to fit anything):\n")
            w(md_table(res["cases"]) + "\n")
            w("A **false positive** is a classifier outcome (occurrence predicted at the notebook's threshold, not quantified); it is explained by the classifier's SHAP plots. "
              "A **concentration error** is a regressor outcome on quantified samples; it is explained by the regressor's SHAP plots. They are different models.\n")
    w("## 5. Risk thresholds and classes (paper 3.15)\n")
    w("**" + thr.provenance + "**\n")
    w("```json\n" + json.dumps(thr.to_dict(), indent=2, default=float) + "\n```\n")
    w("No external SMX benchmark was found in the repository/notebook, so none is applied. Literature PNEC values exist but are matrix- and purpose-specific; "
      "if you adopt one, pass it via `RunConfig.benchmarks={matrix: {'ng_L':..., 'source':..., 'jurisdiction':...}}` and record its source.\n")
    w("Rules (implemented in `smx_uq/risk.py`): low p -> Class I only if the interval is narrow and the conditional concentration is not high; "
      "low p with a wide interval, OOD inputs or high conditional concentration -> Class II (insufficient evidence is **not** low risk; set `uncertain_low_class='I'` to follow the starting table literally); "
      "moderate p -> II (III if concentration is high); high p -> III with moderate concentration, IV with high; high p with low concentration -> II, "
      "or III if the interval is wide and its upper bound reaches a higher level; missing probability or >50% missing inputs -> Unclassified. "
      "Wide intervals add a confirmatory-sampling flag. Classes use predictions only.\n")
    w("Held-out risk-class counts:\n"); w(md_table(res["risk_dist"], index=True) + "\n")
    w("Uncertainty categories (held-out):\n"); w(md_table(res["unc_dist"], index=True) + "\n")
    s = res["sens"]
    if len(s):
        w("Threshold sensitivity (all combinations of the grid in `tables/risk_threshold_sensitivity.csv`): "
          f"agreement with the baseline classes ranges {s['agreement_with_baseline'].min():.2f} to {s['agreement_with_baseline'].max():.2f}; "
          "class shares " + ", ".join(f"{k} {s['share_'+k].min():.2f}-{s['share_'+k].max():.2f}" for k in ["I", "II", "III", "IV"]) + ".\n")
    w("Post-hoc check on test countries (descriptive only, not used for any choice):\n"); w(md_table(res["evalc"]) + "\n")
    w("## 6. Outputs\n")
    w("\n".join(f"* `{f}`" for f in res["files"]) + "\n")
    if res["maps"]:
        w(f"Maps show **sampled sites only** ({res['maps'].get('sites_with_valid_coordinates', 0)} with valid coordinates); no interpolation; held-out and training-country sites are marked differently. "
          "Training-country predictions are in-sample for the classifier/regressor and are not evidence of accuracy.\n")
    w("## 7. Notes, limitations and next steps\n")
    w("\n".join(f"* {n}" for n in res["notes"]) or "* none")
    w("\n* Few countries in validation/test make every country-level statement noisy; coverage and calibration should be re-checked with leave-one-country-out cross-conformal as data grow.\n"
      "* Predictions for unseen countries assume similar covariate relationships; OOD flags are heuristic.\n"
      "* Hyperparameters and model selection used the validation countries, so conformal residuals on them are slightly optimistic.\n"
      "* SHAP explains the fitted model, is affected by correlated predictors, and is not causal. Missing values were median-imputed by the notebook's preprocessing, so dependence plots omit those rows.\n")
    (out / "SMX_UQ_SHAP_RISK_REPORT.md").write_text("\n".join(L))
