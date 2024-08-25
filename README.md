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
You can download the `example_data.csv` file included in this repository to test the model. After running the script with the example data, you can expect to see results similar to the following table:
To generate the following results, run the script using the command:

```bash
python sur_model.py
```
| Variable     | beta_fgls | sterr_fgls | zstat_fgls | prob_fgls     | conf_int_low_95 | conf_int_high_95 |
|--------------|-----------|------------|------------|---------------|-----------------|------------------|
| constant_eq1 | 1.624794  | 0.154010   | 10.549935  | 5.083202e-26  | 1.322940        | 1.926647         |
| x1_1         | 0.818059  | 0.033496   | 24.422598  | 9.841625e-132 | 0.752408        | 0.883710         |
| constant_eq2 | 1.421725  | 0.474635   | 2.995403   | 2.740821e-03  | 0.491456        | 2.351993         |
| x1_2         | 0.177239  | 0.104668   | 1.693347   | 9.038941e-02  | -0.027906       | 0.382383         |
| constant_eq3 | 4.169283  | 0.487859   | 8.546079   | 1.273410e-17  | 3.213096        | 5.125469         |
| x1_3         | 0.018894  | 0.041324   | 0.457218   | 6.475142e-01  | -0.062099       | 0.099887         |
| constant_eq4 | 4.222818  | 0.488548   | 8.643612   | 5.446322e-18  | 3.265282        | 5.180354         |
| x1_4         | 0.001267  | 0.043689   | 0.029007   | 9.768593e-01  | -0.084362       | 0.086897         |
