import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import beta, triang
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    std_likely = (likely_val - min_val) / (max_val - min_val)
    mean = std_likely
    variance = ((max_val - likely_val) * (likely_val - min_val)) / ((max_val - min_val) ** 2 * 12)
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


def calculate_cdf_function(distribution, min_val, max_val, likely_val, x):
    """
    Calculates the CDF based on the specified distribution.

    :param distribution: 'triangular' or 'beta'.
    :param min_val: Minimum value.
    :param max_val: Maximum value.
    :param likely_val: Most likely value.
    :param x: Value to calculate CDF at.
    :return: CDF value.
    """
    if distribution == 'beta':
        return calculate_cdf_beta(min_val, max_val, likely_val, x)
    else:
        return calculate_cdf_triangular(min_val, max_val, likely_val, x)



def plot_joint_distribution(df, asset_column, significant_index, distribution):
    """
    Plots the joint distribution of the asset value and the most significant index.

    :param df: DataFrame with asset and index values.
    :param asset_column: Name of the column with asset values.
    :param significant_index: Name of the most significant index.
    :param distribution: Type of distribution ('triangular' or 'beta').
    """
    # Extract data for Value and significant index
    value_data = df[asset_column]
    index_data = df[significant_index]

    # Generate value and index ranges for plotting
    value_range = np.linspace(value_data['min'], value_data['max'], 100)
    index_range = np.linspace(index_data['min'], index_data['max'], 100)

    # Calculate CDF values for the ranges
    value_cdf = [calculate_cdf_function(distribution, value_data['min'], value_data['max'], value_data['likely'], v)
                 for v in value_range]
    index_cdf = [calculate_cdf_function(distribution, index_data['min'], index_data['max'], index_data['likely'], i)
                 for i in index_range]

    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(value_range, index_range)
    Z = np.outer(value_cdf, index_cdf)

    # Plotting the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Value')
    ax.set_ylabel(significant_index)
    ax.set_zlabel('Joint CDF')

    plt.title(f'Joint Distribution of Value and {significant_index}')
    plt.show()


def estimate_value_expert_mode(df, asset_column, min_row='min', max_row='max', likely_row='likely',
                               distribution='triangular', scale_data=False, plot_joint_dist=True):
    """
    Estimates the asset value based on a DataFrame containing min, max, and most likely values for each index.
    Optionally plots the joint distribution with the most significant index.
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

    # Ensure index_vals is 2D and asset_val is 1D
    if index_vals.ndim == 1:
        index_vals = index_vals.reshape(-1, 1)
    asset_val = np.array([asset_val])

    # Scale the index values if scale_data is True
    if scale_data:
        scaler = StandardScaler()
        index_vals = scaler.fit_transform(index_vals)

    # NNLS Regression to find weights
    weights, _ = nnls(index_vals, asset_val)

    # Identify the most significant index based on weights
    significant_index = df.columns[index_columns[np.argmax(weights)]]

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

    # Combine CDFs to estimate the final value
    joint_cdf = np.prod(cdf_values)

    # Inverse transformation to get the asset value in original scale
    min_asset, max_asset = min_vals[asset_idx], max_vals[asset_idx]
    estimated_value = joint_cdf * (max_asset - min_asset) + min_asset

    # Plot joint distribution if required
    if plot_joint_dist:
        plot_joint_distribution(df, asset_column, significant_index, distribution)

    return estimated_value


# Example usage of the function
data = {
    'Value': [40000, 300000, 250000],
    'Soil Quality': [2, 8, 6],
    'Water Availability': [3, 15, 8],
    'Accessibility': [5, 13, 7]
}
df = pd.DataFrame(data, index=['min', 'max', 'likely'])

# Run the valuation function and plot joint distribution
estimated_value = estimate_value_expert_mode(df, asset_column='Value', distribution='triangular')

print(f'The estimated price of the asset is {estimated_value}')