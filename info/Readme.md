# Bicycle Relocation Optimization

This project solves the RentalBike bicycle-relocation problem and exports the
solver-proven plan to a business-ready Excel workbook.

## Setup

Use Python 3.12 from the repository root:

```powershell
python -m pip install -r info/requirements.txt
jupyter lab
```

Open `main.ipynb` and run all cells in order.

## Tests

Run the seven model and workbook regression tests from the repository root:

```powershell
python -m pytest
```

The suite executes the actual data-driven model cells from `main.ipynb`, solves
the MILP once, and validates its structure, feasibility, known optimum, objective
reconciliation, and exported workbook.

## Workbook contract

- Input: `data/BicyclesRelocationData.xlsx`
- Output: `results/BicyclesRelocationData_Optimized.xlsx`
- Generated worksheet: `OptimalRelocationPlan`

The input workbook is read-only and is never overwritten. Each run rebuilds the
output workbook from the input, preserves every original worksheet, and adds the
generated `OptimalRelocationPlan` worksheet. If the output workbook is open in
Excel, close it before rerunning the export cell.

## Project files

- `main.ipynb`: complete analysis, optimization, validation, and export workflow
- `tests/pytest_review.py`: pytest review of the notebook model and Excel output
- `pytest.ini`: test discovery for the requested `pytest_review.py` filename
- `AI/AI_assistance_review.ipynb`: AI-assistance review copy of the notebook
- `info/case-exact-model.pdf`: case specification
- `Presentation/Business_case.pptx`: business presentation
- `info/structure.txt`: expected deliverable layout
