# Project Scope

## Goal

The goal of this project is to complete the CMOR 438 / INDE 577 final project: a
self-contained machine learning study that applies algorithms covered in the
course to a chosen dataset, and clearly communicates the methodology, results,
and interpretation.

## Required Deliverables

- A public repository containing all project code and documentation.
- A custom Python package (`cmor438_ml`) holding reusable implementations and
  helper utilities.
- Jupyter notebooks covering data exploration, preprocessing, modeling,
  evaluation, and interpretation.
- A clear README describing the project, structure, and how to reproduce the
  work.
- Reproducible setup via `requirements.txt` and `pyproject.toml`.
- A test suite that verifies the package can be imported and used.

## Grading Criteria

The project will be assessed against the following weighting:

| Criterion                          | Weight |
| ---------------------------------- | ------ |
| Functionality and Implementation   | 40%    |
| Documentation and Readability      | 20%    |
| Testing and Reliability            | 20%    |
| Examples and Usability             | 10%    |
| Repository Quality                 | 10%    |

## Current Setup Decision

For the initial stage, the project uses a `src/` layout with a single installable
package (`cmor438_ml`), `setuptools` for packaging, and `pytest` for testing.
Dependencies are pinned by name (unversioned) in `requirements.txt` and the
project is installed in editable mode during development. This keeps the
skeleton minimal while leaving room to grow.

## Items Still to Decide

The following decisions are deferred to later stages:

- **Dataset choice** — which dataset(s) to use, their source, size, and license.
- **Algorithm scope** — which supervised and unsupervised algorithms to include,
  and whether they will be implemented from scratch, used via libraries, or both.
- **Evaluation strategy** — metrics, validation scheme, and baselines.
- **Notebook breakdown** — how the analysis is split across notebooks.
- **Reporting format** — structure and depth of the final written conclusions.
- **Dependency versions** — whether to pin exact versions before submission.
