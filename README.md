# Seemingly Unrelated Regression (SUR) Model Implementation

This repository contains a Python implementation of the Seemingly Unrelated Regression (SUR) model. The code is designed to estimate parameters in a system of equations where the errors across different equations are correlated.

## Project Overview

The SUR model is a generalization of the ordinary least squares (OLS) method, used when you have multiple regression equations that may have correlated error terms. This implementation computes the Feasible Generalized Least Squares (FGLS) estimators, which account for these correlations, providing more efficient parameter estimates.

## CSV Format

The input data for the SUR model should be provided in a CSV file with the following format:
- Columns for the independent variables (X) of all individuals are grouped by variable.
- For `n` individuals with `m` independent variables each, the format is:

## Example for 2 individuals, 3 independent variables:

### Table Representation

| x1_1 | x1_2 | x2_1 | x2_2 | x3_1 | x3_2 | y1  | y2  |
|------|------|------|------|------|------|-----|-----|
| ...  | ...  | ...  | ...  | ...  | ...  | ... | ... |

## Getting Started

### Prerequisites

Ensure that you have the following Python libraries installed:
- `pandas`
- `numpy`
- `scipy`

You can install these dependencies using pip:

```bash
pip install pandas numpy scipy
```
## Example output
Final results table:
| Variable         | Beta (F)GLS | Std. Error | Z-Statistic | P-Value | 95% CI Low | 95% CI High |
|------------------|-------------|------------|-------------|---------|------------|-------------|
| constant_eq1     | ...         | ...        | ...         | ...     | ...        | ...         |
| independent_var1 | ...         | ...        | ...         | ...     | ...        | ...         |
| ...              | ...         | ...        | ...         | ...     | ...        | ...         |

