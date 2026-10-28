# SMX uncertainty / SHAP / risk-class add-on

**Status: implemented and unit-tested on synthetic data only. Not yet run on the real SMX data** (the uploaded notebook reads
`/kaggle/input/.../SMX_occurrence.csv` and `SMX_concentration.csv`, which were not provided; no saved models either). No empirical SMX results exist yet.

## Run
1. Open `smxensemble_uq.ipynb` on Kaggle (original 52 cells unchanged) -> *Run All*. The 5 appended cells install `shap folium`, write the `smx_uq` package, build the context from the notebook's objects and run everything.
2. Outputs: `/kaggle/working/smx_uq_outputs/{tables,figures,maps,models}` + `SMX_UQ_SHAP_RISK_REPORT.md` (filled with the real numbers from that run).
3. Tests: `cd smx_uq_package && python -m unittest discover -s tests -v` (or `pytest tests`).

## Plan implemented
Refit selected tuned tree models on train countries -> calibrate (none/Platt/isotonic chosen by country-grouped cross-fitted log loss on validation) ->
normalised split-conformal 90 % intervals on log10(C+1) (scale model on training-country OOF residuals; quantile from validation countries) ->
OOD flags (training ranges) -> provisional thresholds (training/validation only) -> 4-class risk + uncertainty categories + sensitivity grid ->
TreeSHAP on held-out countries (global, direction, dependence, interactions, local, site table) -> maps of sampled sites -> report.
