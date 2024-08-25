# Importing necessary libraries
import pandas as pd
import numpy as np
from numpy.linalg import inv
# CSV Format for the SUR Model:
# - Columns for x variables of all individuals are grouped by variable.
# - For 'n' individuals with 'm' x variables each, format is:
#   x1_1, x1_2, ..., x1_n, x2_1, x2_1, ..., x2_n, ..., xm_1, ..., xm_n, y1, y2, ..., yn.
# - Example for 2 individuals, 3 x variables:
#   x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, y1, y2.

# Table Representation for the aforementioned example:
# +------+------+------+------+------+------+-----+-----+
# | x1_1 | x1_2 | x2_1 | x2_2 | x3_1 | x3_2 | y1  | y2  |
# +------+------+------+------+------+------+-----+-----+
# |      |      |      |      |      |      |     |     |
# +------+------+------+------+------+------+-----+-----+
# (More rows follow...)

# Load CSV into DataFrame
# Note: if you opt for the "header = None" argument in the read_csv function, then you dont need to add column names (i.e., x1_1, x1_2)
df = pd.read_csv("Date_panel_SUR - 4indv - doua var.csv") # Change the path according to your directory
# Display first few rows
df.head()
# Prompt for the number of individuals
num_individuals = int(input("Enter the number of individuals: "))  # Converts the input to an integer

# Prompt for the number of variables per individual
# The input is expected to be numbers separated by spaces, e.g., "2 2 2"
num_variables_per_individuals = list(map(int, input("Enter the number of variables per individual, separated by spaces: ").split()))

# Automatically detect the number of available observations
num_observations = df.shape[0]

# Initialize dictionaries to store OLS estimators and residuals for each individual
ols_estimators = {}
residuals = {}
X_blocks = []  # To store X matrices for each individual
Y_vectors = [] # To store Y vectors for each individual

# Assuming the last 'num_individuals' columns are the Y's
Y_indices = list(range(-num_individuals, 0))

# Loop to process each individual
for i in range(num_individuals):
    # Columns for X's of individual 'i'
    X_indices = [i + num_individuals * j for j in range(num_variables_per_individuals[i])]
    X = df.iloc[:, X_indices].values
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    X_blocks.append(X)

    # Extracting the Y for individual 'i'
    Y = df.iloc[:, Y_indices[i]].values
    Y_vectors.append(Y)

    # Compute OLS estimator and residuals
    beta_hat = inv(X.T @ X) @ X.T @ Y
    ols_estimators[f'Individual {i+1}'] = beta_hat
    residuals[f'Individual {i+1}'] = Y - X @ beta_hat

# Display the OLS estimators for each individual
for individual in ols_estimators:
    print(f'{individual} OLS Estimators: {ols_estimators[individual]}')

# Convert the dictionary of residuals into a list of arrays (one for each individual)
residuals_list = [residuals[key] for key in residuals]

# Convert the list of residuals into a NumPy array
residuals_array = np.array(residuals_list)

# Compute the variance-covariance matrix between the residuals
sigma = np.cov(residuals_array)

# Print the variance-covariance matrix
print(sigma)
# Create a function for the block diagonal matrix
def manual_block_diag(blocks):
    # Determine the size of the final matrix
    total_rows = sum(block.shape[0] for block in blocks)
    total_cols = sum(block.shape[1] for block in blocks)

    # Initialize a matrix of zeros
    result = np.zeros((total_rows, total_cols))

    # Place each block in the result matrix
    current_row = 0
    current_col = 0
    for block in blocks:
        rows, cols = block.shape
        result[current_row:current_row + rows, current_col:current_col + cols] = block
        current_row += rows
        current_col += cols

    return result

# Create the X_sur matrix using the above function
X_sur = manual_block_diag(X_blocks)

# Concatenate Y vectors to create Y_sur
Y_sur = np.concatenate(Y_vectors)
# Create the Omega matrix
# Omega is a block diagonal matrix representing the variance-covariance structure of errors across individuals.
# Its dimension is the total number of observations across all individuals (num_observations*num_individual)
Omega = np.zeros((num_observations * num_individuals, num_observations * num_individuals))

# Fill in the Omega matrix
for i in range(num_individuals):
    for j in range(num_individuals):
        # Fill the diagonal blocks of Omega. Each block corresponds to the covariance between individuals i and j.
        # The diagonal blocks (where i == j) represent the variance of the errors for each individual.
        np.fill_diagonal(Omega[num_observations * i:num_observations * (i + 1), num_observations * j:num_observations * (j + 1)], sigma[i, j])
# Compute the FGLS (Feasible Generalized Least Squares) estimators
# FGLS is used in SUR models to account for the correlation between error terms of different equations.
Omega_inv = inv(Omega)
beta_fgls = inv(X_sur.T @ Omega_inv @ X_sur) @ (X_sur.T @ Omega_inv @ Y_sur)

print("Beta (F)GLS:")
print(beta_fgls)
from scipy.stats import norm
# Compute Asymptotic Variance (AsyVar)
AsyVar = inv(X_sur.T @ inv(Omega) @ X_sur)

# Print AsyVar
#print("AsyVar:")
#print(AsyVar)
# Calculate standard errors (sterr_fgls)
sterr_fgls = np.sqrt(np.diag(AsyVar))

# Calculate Z-statistics (zstat_fgls)
zstat_fgls = beta_fgls / sterr_fgls

# Compute p-values (prob_fgls) for the two-tailed test
prob_fgls = norm.cdf(-np.abs(zstat_fgls)) * 2

# Compute the 95% confidence intervals
# The critical value for a 95% confidence level is approximately 1.96 for a two-tailed test
critical_value = norm.ppf(0.975)  # This returns the inverse cdf for the value 0.975
conf_int_low_95 = beta_fgls - critical_value * sterr_fgls
conf_int_high_95 = beta_fgls + critical_value * sterr_fgls
# Printing options
pd.set_option('display.max_columns', None)  # This will force pandas to display all columns
pd.set_option('display.expand_frame_repr', False)  # This prevents the DataFrame from being split across the terminal width

# Print the results in a table-like format
result_df = pd.DataFrame(
    np.column_stack((beta_fgls, sterr_fgls, zstat_fgls, prob_fgls, conf_int_low_95, conf_int_high_95)),
    columns=['beta_fgls', 'sterr_fgls', 'zstat_fgls', 'prob_fgls', 'conf_int_low_95', 'conf_int_high_95']
)

# Given df is your DataFrame with headers as variable names
# Assuming the last num_individuals columns are the dependent variables, and the rest are independent variables
independent_vars = df.columns[:-num_individuals]

# Number of coefficients for each equation, assuming each has a constant term
num_coeffs_per_eq = len(independent_vars) // num_individuals + 1

# Construct the variable names list for the FGLS output
variable_names = []
for i in range(num_individuals):
    variable_names.append(f'constant_eq{i+1}')  # Append constant with equation number
    # Add the independent variable names, stripping any pandas-added numerical suffixes
    variable_names.extend([var.split('.')[0] for var in independent_vars[i::num_individuals]])

# Now, attach these names to your FGLS results DataFrame
result_df.index = variable_names[:len(beta_fgls)]  # Ensure the length matches the number of coefficients

# Display the DataFrame with the new index (variable names)
print(result_df)