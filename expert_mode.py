import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import beta, triang
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def standardize_values(min_val, max_val, likely_val):
    """
    Standardizes the values to a [0, 1] scale.
    """
    return (likely_val - min_val) / (max_val - min_val)


def calculate_cdf_triangular(min_val, max_val, likely_val, x):
    """
    Calculate the CDF for triangular distribution at point x.
    """
    c = (likely_val - min_val) / (max_val - min_val)
    return triang.cdf(x, c, loc=min_val, scale=max_val - min_val)

def estimate_beta_parameters(min_val, max_val, likely_val):
    """
    Estimate the parameters of the beta distribution (alpha and beta)
    based on the min, max, and likely (mode) values.
    """
    # Standardize values to [0, 1]
    std_likely = (likely_val - min_val) / (max_val - min_val)

    # Estimating alpha and beta using method of moments
    # Note: This is a simple estimation method and might not be the best fit for all cases
    mean = std_likely
    variance = ((max_val - likely_val) * (likely_val - min_val)) / ((max_val - min_val)**2 * 12)

    # Solving for alpha and beta
    temp = mean * (1 - mean) / variance - 1
    alpha = mean * temp
    beta_param = (1 - mean) * temp

    return alpha, beta_param

def calculate_cdf_beta(min_val, max_val, likely_val, x):
    """
    Calculate the CDF for beta distribution at point x.
    """
    a, b = estimate_beta_parameters(min_val, max_val, likely_val)
    return beta.cdf((x - min_val) / (max_val - min_val), a, b)


def estimate_value_expert_mode(df, asset_column, min_row='min', max_row='max', likely_row='likely',
                               distribution='triangular', scale_data=False):
    """
    Estimates the asset value based on a DataFrame containing min, max, and most likely values for each index.

    :param df: DataFrame containing the values.
    :param asset_column: Name or number of the column containing the asset values.
    :param min_row: Row label for minimum values.
    :param max_row: Row label for maximum values.
    :param likely_row: Row label for most likely values.
    :param distribution: Type of distribution ('triangular' or 'beta').
    :return: Estimated value of the asset.
    """
    # Extracting values from the DataFrame
    min_vals = df.loc[min_row].values
    max_vals = df.loc[max_row].values
    likely_vals = df.loc[likely_row].values

    # Identifying asset and index columns
    asset_idx = df.columns.get_loc(asset_column)
    index_columns = [i for i in range(df.shape[1]) if i != asset_idx]

    if not index_columns:
        raise ValueError("No index columns provided.")

    # Standardize the values
    standardized_vals = np.array([standardize_values(min_val, max_val, likely_val)
                                  for min_val, max_val, likely_val in zip(min_vals, max_vals, likely_vals)])

    # Prepare data for NNLS
    asset_val = standardized_vals[asset_idx]
    index_vals = np.vstack([standardized_vals[i] for i in index_columns]).T

    # Ensure index_vals is 2D
    if index_vals.ndim == 1:
        index_vals = index_vals.reshape(-1, 1)

    # Ensure asset_val is 1D
    asset_val = np.array([asset_val])

    # Scale the index values if scale_data is True
    if scale_data:
        scaler = StandardScaler()
        index_vals = scaler.fit_transform(index_vals)

    # NNLS Regression to find weights
    weights, _ = nnls(index_vals, asset_val)
    print("Weights:", weights)  # Debugging: print the weights

    # Calculate the CDFs using the estimated weights
    cdf_values = []
    for i, weight in zip(index_columns, weights):
        min_val, max_val, likely_val = min_vals[i], max_vals[i], likely_vals[i]
        # Using the midpoint of min and max as the sample point for CDF calculation
        x = (min_val + max_val) / 2
        cdf_val = calculate_cdf_beta(min_val, max_val, likely_val,
                                     x) if distribution == 'beta' else calculate_cdf_triangular(min_val, max_val,
                                                                                                likely_val, x)
        weighted_cdf = cdf_val ** weight
        cdf_values.append(weighted_cdf)
        print(f"CDF for index {i}: {cdf_val}, Weighted CDF: {weighted_cdf}")  # Debugging: print CDFs

    # Combine CDFs to estimate the final value
    joint_cdf = np.prod(cdf_values)
    print("Joint CDF:", joint_cdf)  # Debugging: print the joint CDF

    # Inverse transformation to get the asset value in original scale
    min_asset, max_asset = min_vals[asset_idx], max_vals[asset_idx]
    estimated_value = joint_cdf * (max_asset - min_asset) + min_asset

    # Preparing x_values for CDF calculation
    x_values = np.linspace(0, 1, 100)

    # Updated CDF calculation
    cdf_values_dict = {}
    for i, weight in zip(index_columns, weights):
        min_val, max_val, likely_val = min_vals[i], max_vals[i], likely_vals[i]
        cdf_values = []
        for x in x_values:
            cdf_val = calculate_cdf_beta(min_val, max_val, likely_val,
                                         x) if distribution == 'beta' else calculate_cdf_triangular(min_val, max_val,
                                                                                                    likely_val, x)
            cdf_values.append(cdf_val ** weight)
        cdf_values_dict[df.columns[i]] = cdf_values

    # Return the estimated value, weights, and CDF values
    return estimated_value, weights, cdf_values_dict


def plot_weights(weights, index_names):
    plt.bar(index_names, weights)
    plt.xlabel('Index')
    plt.ylabel('Weight')
    plt.title('Weights Assigned to Each Index')
    plt.show()


def plot_cdfs(cdf_values_dict, x_values):
    for index, cdf_values in cdf_values_dict.items():
        plt.plot(x_values, cdf_values, label=index)
    plt.xlabel('Standardized Value')
    plt.ylabel('CDF Value')
    plt.title('CDFs of Each Index')
    plt.legend()
    plt.show()


data = {
    'Value': [40000, 800000, 250000],  # Keep the asset values as is
    'Soil Quality': [20, 38, 22],         # Widen the range and adjust the likely value
    'Water Availability': [4, 30, 25], # Adjust the likely value closer to the maximum
    'Accessibility': [5, 13, 7],     # Adjust the likely value closer to the maximum
}

df = pd.DataFrame(data, index=['min', 'max', 'likely'])

# Run the valuation function
estimated_value, weights, cdf_values_dict = estimate_value_expert_mode(df, asset_column='Value',
                                                                       distribution='triangular')

# Plotting
index_names = [col for col in df.columns if col != 'Value']
plot_weights(weights, index_names)

# Generating x values for CDF plots (standardized scale)
x_values = np.linspace(0, 1, 100)
plot_cdfs(cdf_values_dict, x_values)

print(f"Estimated Asset Value: ${estimated_value:.2f}")



