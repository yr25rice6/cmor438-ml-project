# Examples

Small, focused scripts showing how to use the custom `cmor438_ml` package
outside of the full analysis notebook.

## Available examples

- **`basic_classification_demo.py`** — a lightweight, terminal-only example of
  the end-to-end workflow. It loads the Breast Cancer Wisconsin (Diagnostic)
  dataset, makes a train/test split, fits `StandardScalerScratch` on the
  training set, trains `KNNClassifier` and `LogisticRegressionGD`, evaluates both
  with the package metrics (accuracy, precision, recall, F1), reports a ROC/AUC
  summary for the logistic regression model, and prints a short readable summary.
  It produces no plots and writes no files.

## Running

From the repository root (Windows PowerShell), with the virtual environment set
up as described in the top-level `README.md`:

```powershell
.\.venv\Scripts\python.exe examples/basic_classification_demo.py
```
