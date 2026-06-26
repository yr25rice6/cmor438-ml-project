# Notebooks

This directory holds the Jupyter notebooks for the project. It contains the main
end-to-end workflow notebook along with dedicated per-algorithm directories that
follow the structure of the course example repositories.

## Main workflow notebook

- **`01_breast_cancer_classification_workflow.ipynb`** — the main analysis
  notebook. It demonstrates the end-to-end binary classification workflow for
  the Breast Cancer Wisconsin dataset: loading the data, a brief exploratory
  analysis, a train/test split, feature standardization, training the KNN and
  logistic regression models, evaluating and comparing the two models, a
  five-fold cross-validation check on that comparison, a short KNN
  hyperparameter check, and a ROC/AUC analysis of the logistic regression model
  (treating benign as the positive class).

## Algorithm directories

The per-algorithm material is organized into two directories:

- **`Supervised/`** — dedicated notebooks for the supervised learning
  algorithms (see `Supervised/README.md`).
- **`Unsupervised/`** — dedicated notebooks for the unsupervised learning
  algorithms (see `Unsupervised/README.md`).

## Dedicated algorithm notebooks

The first six supervised notebooks are available now. The remaining
directories exist as placeholders (each holds a `.gitkeep` file); those
notebooks will be added in a later step.

### Supervised

1. **Perceptron** *(available)* — `Supervised/1. Perceptron/perceptron.ipynb`
2. **Gradient Descent** *(available)* — `Supervised/2. Gradient Descent/gradient_descent.ipynb`
3. **Linear Regression** *(available)* — `Supervised/3. Linear Regression/linear_regression.ipynb`
4. **Logistic Regression** *(available)* — `Supervised/4. Logistic Regression/logistic_regression.ipynb`
5. **Multilayer Perceptron** *(available)* — `Supervised/5. Multilayer Perceptron/multilayer_perceptron.ipynb`
6. **K Nearest Neighbors** *(available)* — `Supervised/6. K Nearest Neighbors/k_nearest_neighbors.ipynb`
7. **Decision Tree** *(planned)* — `Supervised/7. Decision Tree/`
8. **Random Forest** *(planned)* — `Supervised/8. Random Forest/`

### Unsupervised

- **K Means Clustering** — `Unsupervised/K Means Clustering/`
