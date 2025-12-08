# Binary Anomaly Streamlit App

A Streamlit UI for loading tabular binary datasets, running cross‑validated models (RandomForest / CatBoost / EBM), visualizing EBM interactions with Plotly, mining rules, and checking drift.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate on *nix
pip install -r requirements.txt   # core deps

# Optional extras (enable SHAP, RuleFit, Evidently, Feather I/O):
pip install shap rulefit "evidently>=0.4,<0.5" "pyarrow>=14,<15"

streamlit run app.py
```

## Sample data
- CSV: `data/sample_binary.csv`
- Pickle: `data/sample_binary.pkl`
- Feather: `data/sample_binary.feather` (create by installing `pyarrow`, then re-saving)

## Usage
1. In the sidebar, choose a working directory + file (csv/feather/pkl/pickle) or upload a file.
2. Pick target column and training features (filter + multiselect). Adjust class weights and CV folds.
3. Choose a model (RandomForest, CatBoost, EBM) and click **Run CV Training**.
4. Tabs:
   - Data Overview, Univariate Dist., Pairwise Relations (EBM main/interaction Plotly), Model & CV, Rules/Interactions.
   - Drift & Summary: requires a `TYPE` column with values `TRAIN`, `VALID`, `TEST` and Evidently installed. Reference split defaults to `TRAIN`; compare against `VALID`/`TEST` with optional row sampling.

## Notes
- Optional packages: `shap` for tree SHAP importance, `rulefit` for rule mining, `evidently` for drift, `pyarrow` for Feather support.
- If performance is a concern for drift reports, keep the sampling slider low; Evidently can be heavy on large frames.
