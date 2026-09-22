"""Generate notebooks/01_baselines.ipynb, then execute it in place so it
carries real outputs (not just unexecuted code)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Baseline models on the Crunchbase startup outcome dataset\n\n"
    "Loads the models trained by `python -m src.models.train` and reports "
    "their saved CV and held-out test metrics side by side. Run "
    "`python -m src.models.train` first if `models/` is empty."
))

cells.append(nbf.v4.new_code_cell(
    "import json\n"
    "from pathlib import Path\n\n"
    "import pandas as pd\n\n"
    "MODEL_NAMES = [\n"
    "    'dummy_most_frequent',\n"
    "    'logistic_regression_full',\n"
    "    'logistic_regression_clean',\n"
    "    'hist_gradient_boosting_full',\n"
    "    'hist_gradient_boosting_clean',\n"
    "]\n\n"
    "metadata = {name: json.loads(Path(f'../models/{name}_metadata.json').read_text()) for name in MODEL_NAMES}\n"
    "len(metadata)"
))

cells.append(nbf.v4.new_markdown_cell("## Metrics table (test = time-separated holdout, CV = repeated stratified CV on train)"))

cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for name, m in metadata.items():\n"
    "    rows.append({\n"
    "        'model': name,\n"
    "        'feature_set': m['feature_set'],\n"
    "        'cv_roc_auc_mean': m['cv']['roc_auc_mean'],\n"
    "        'cv_roc_auc_std': m['cv']['roc_auc_std'],\n"
    "        'test_roc_auc': m['test']['roc_auc'],\n"
    "        'test_average_precision': m['test']['average_precision'],\n"
    "        'test_brier': m['test']['brier_score'],\n"
    "        'threshold': m['threshold'],\n"
    "        'test_f0.5': m['test']['f0.5'],\n"
    "    })\n"
    "table = pd.DataFrame(rows).set_index('model')\n"
    "table"
))

cells.append(nbf.v4.new_markdown_cell(
    "## CV vs. test: why they differ\n\n"
    "CV metrics are computed on repeated stratified folds *within* the train cohort "
    "(same era, positive rate 59.83%). The test metric is a single evaluation on the "
    "time-separated holdout (a later era, positive rate 26.71%). The gap between the "
    "two numbers reflects genuine distribution shift across time, not just sampling noise - "
    "this is the realistic deployment scenario (scoring newer companies), so the test number "
    "is the one that matters for expectations, not the CV number."
))

cells.append(nbf.v4.new_code_cell(
    "table[['cv_roc_auc_mean', 'test_roc_auc']].assign(\n"
    "    gap=lambda d: d['test_roc_auc'] - d['cv_roc_auc_mean']\n"
    ")"
))

cells.append(nbf.v4.new_markdown_cell("## Clean vs. full: the leakage finding"))

cells.append(nbf.v4.new_code_cell(
    "for family in ['logistic_regression', 'hist_gradient_boosting']:\n"
    "    full_auc = metadata[f'{family}_full']['test']['roc_auc']\n"
    "    clean_auc = metadata[f'{family}_clean']['test']['roc_auc']\n"
    "    print(f'{family}: full={full_auc:.4f} clean={clean_auc:.4f} gap={full_auc - clean_auc:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Figures\n\n"
    "See `../reports/figures/roc_comparison.png`, `pr_comparison.png`, "
    "`calibration_comparison.png` for the full ROC/PR/calibration curves across all 5 models."
))

nb["cells"] = cells

path_out = "notebooks/01_baselines.ipynb"
with open(path_out, "w") as f:
    nbf.write(nb, f)

client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
client.execute()
with open(path_out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote and executed {path_out}")
