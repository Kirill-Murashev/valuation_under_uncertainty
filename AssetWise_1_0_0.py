'''
Cyrill A. Murashev, Sovconsult DOO, Montenegro, Herzeg Novi.
This code implements the method described in the paper
"A generalized method for valuing agricultural farms under uncertainty".
C. García, J. García, M.M. Lópezc, R. Salmerón
https://www.sci-hub.ru/10.1016/j.landusepol.2017.04.008.
This version implements the "expert mode" of the method, in which the appraiser uses his judgment about
on the minimum, maximum and probable values of the parameters, including the value.
'''

import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import beta, triang
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage


def load_data(file_path):
    """
    Loads data from an XLSX file into a DataFrame.
    """
    return pd.read_excel(file_path, index_col=0)


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
    value_range = np.linspace(value_data['minimum'], value_data['maximum'], 100)
    index_range = np.linspace(index_data['minimum'], index_data['maximum'], 100)

    # Calculate CDF values for the ranges
    value_cdf = [calculate_cdf_function(distribution, value_data['minimum'],
                                        value_data['maximum'],
                                        value_data['likely'], v)
                 for v in value_range]
    index_cdf = [calculate_cdf_function(distribution, index_data['minimum'],
                                        index_data['maximum'],
                                        index_data['likely'], i)
                 for i in index_range]

    # Create meshgrid for 3D plot
    x, y = np.meshgrid(value_range, index_range)
    z = np.outer(value_cdf, index_cdf)

    # Plotting the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel('Value')
    ax.set_ylabel(significant_index)
    ax.set_zlabel('Joint CDF')

    plt.title(f'Joint Distribution of Value and {significant_index}')
    return fig


def save_plots(df, transformation, folder_path):
    """
    Saves the plot to the specified folder.
    """
    fig = plot_joint_distribution(df, 'Value', df.columns[1], 'triangular')
    fig.savefig(os.path.join(folder_path, f'plot_{transformation}.png'))


def estimate_value_expert_mode(df, asset_column, min_row='minimum', max_row='maximum', likely_row='likely',
                               distribution='triangular', scale_data=False, plot_joint_dist=True):
    """
    Estimates the asset value based on a DataFrame containing min, max, and most likely values for each index.
    Additionally, returns weights, individual CDFs, and joint CDF.
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

    # Calculate the CDFs using the estimated weights
    cdf_values = []
    for i, weight in zip(index_columns, weights):
        min_val, max_val, likely_val = min_vals[i], max_vals[i], likely_vals[i]
        x = (min_val + max_val) / 2  # Sample point for CDF calculation
        cdf_val = calculate_cdf_function(distribution, min_val, max_val, likely_val, x)
        weighted_cdf = cdf_val ** weight
        cdf_values.append(weighted_cdf)

    # Combine CDFs to estimate the final value
    joint_cdf = np.prod(cdf_values)

    # Inverse transformation to get the asset value in original scale
    min_asset, max_asset = min_vals[asset_idx], max_vals[asset_idx]
    estimated_value = joint_cdf * (max_asset - min_asset) + min_asset

    # Plot joint distribution if required
    if plot_joint_dist:
        plot_joint_distribution(df, asset_column, df.columns[index_columns[np.argmax(weights)]], distribution)

    # Return the estimated value, weights, CDF values, and joint CDF
    return {
        "Estimated Value": estimated_value,
        "Weights": weights,
        "Individual CDFs": cdf_values,
        "Joint CDF": joint_cdf
    }


def square(x):
    """
    Calculate the square of a given number.

    Parameters:
        x (int or float): The number to be squared.

    Returns:
        int or float: The square of the given number.
    """
    return x ** 2


def square_root(x):
    """
    Calculates the square root of a given number.

    Parameters:
        x (float): The number to calculate the square root for.

    Returns:
        float: The square root of the given number.
    """
    return np.sqrt(x)


def logarithm(x):
    """
    Calculate the natural logarithm of a number.

    Parameters:
        x (float): The number to calculate the logarithm of.

    Returns:
        float: The natural logarithm of the input number plus 1, to avoid log(0).
    """
    return np.log(x + 1)  # Adding 1 to avoid log(0)


def inverse(x):
    """
    Calculate the inverse of a number.

    Parameters:
        x (float): The number to calculate the inverse of.

    Returns:
        float: The inverse of the input number.

    Notes:
        Adding a small number to the input to avoid division by zero.
    """
    return 1 / (x + 1e-6)  # Adding a small number to avoid division by zero


# Apply transformations to the DataFrame
def apply_transformations(df, exclude_column='Value'):
    """
    Apply transformations to a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to apply transformations to.
        exclude_column (str, optional): The column to exclude from transformations. Defaults to 'Value'.

    Returns:
        DataFrame: The transformed DataFrame.
    """
    transformed_df = df.copy()
    for col in df.columns:
        if col != exclude_column:
            transformed_df[col + '_squared'] = df[col].apply(square)
            transformed_df[col + '_sqrt'] = df[col].apply(square_root)
            transformed_df[col + '_log'] = df[col].apply(logarithm)
    return transformed_df


def apply_transformations_to_value(df, transformation):
    transformed_df = df.copy()
    if transformation == 'squared':
        transformed_df['Value'] = df['Value'] ** 2
    elif transformation == 'sqrt':
        transformed_df['Value'] = np.sqrt(df['Value'])
    elif transformation == 'log':
        transformed_df['Value'] = np.log(df['Value'] + 1)  # Adding 1 to avoid log(0)
    return transformed_df


def main(file_path):
    """
    Main function to process the input file and save the output.
    """
    try:
        df = load_data(file_path)
        transformed_df = apply_transformations(df)

        transformations = ['raw', 'squared', 'sqrt', 'log']
        estimated_values = []

        for transformation in transformations:
            final_df = apply_transformations_to_value(transformed_df, transformation)
            estimation_results = estimate_value_expert_mode(final_df, 'Value', distribution='triangular')
            estimated_value = estimation_results["Estimated Value"]

            if transformation == 'squared':
                estimated_value = np.sqrt(estimated_value)
            elif transformation == 'sqrt':
                estimated_value = estimated_value ** 2
            elif transformation == 'log':
                estimated_value = np.exp(estimated_value) - 1

            save_plots(final_df, transformation, os.path.dirname(file_path))
            estimated_values.append(estimated_value)

        mean_estimated_value = np.mean(estimated_values)
        print(f"The mean estimated value: {round(mean_estimated_value, 2)}")

        # Save results to file
        result_df = pd.DataFrame({'Transformation': transformations + ['Final outcome'],
                                  'Estimated Value': estimated_values + [mean_estimated_value]})
        result_df.to_excel(os.path.join(os.path.dirname(file_path), 'AssetWise_output.xlsx'))

    except Exception as e:
        print(f"An error occurred: {e}")

        # Save results to file
        result_df = pd.DataFrame({'Transformation': transformations, 'Estimated Value': estimated_values})
        result_df.to_excel(os.path.join(os.path.dirname(file_path), 'AssetWise_output.xlsx'))
    except Exception as e:
        print(f"An error occurred: {e}")


def select_file():
    """
    Prompts the user to select a file using a file dialog.

    Returns:
        None
    """
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"), ("Excel Files (older)", "*.xls")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        run_button.config(state=tk.NORMAL)


def run_script():
    """
    Runs a script based on the file path provided.

    Parameters:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the file path provided does not exist.
        Exception: If an error occurs while running the script.

    This function prompts the user to enter a file path.
    If a file path is provided, the function calls the main script function passing the file path as an argument.
    If the script runs successfully, a success message is displayed. If an error occurs, an error message is displayed.
    If no file path is provided, a warning message will appear.
    """
    file_path = file_entry.get()
    if file_path:
        # Here you would call your main script function
        try:
            # Assuming your main function accepts the file path
            main(file_path)
            messagebox.showinfo("Success", "Process completed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("Warning", "Please select a file first")


def show_info():
    info_window = tk.Toplevel(root)
    info_window.title("About AssetWise 1.0.0")

    info_text = """
    AssetWise 1.0.0
    Valuation Tool under Uncertainty

    Copyright 2023 Sovconsult DOO

    For support, contact us at:
    Facebook: https://www.facebook.com/groups/1977067932456703
    Telegram: https://t.me/AIinValuation

    Legal Information:
    [Copyright [2023] [Sovconsult DOO]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.]
    """
    tk.Label(info_window, text=info_text, justify=tk.LEFT, padx=10, pady=10).pack()

    close_button = tk.Button(info_window, text="Close", command=info_window.destroy)
    close_button.pack(pady=10)


# Create the main window
root = tk.Tk()
root.title("AssetWise 1.0.0: Valuation Tool Under Uncertainty")

# Load the .png file as a PhotoImage
icon = PhotoImage(file='/home/kaarlahti/PycharmProjects/valuation_uncertainty/logo.png')

# Set the icon
root.iconphoto(True, icon)

# Create and place widgets
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=10, pady=10)

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.grid(row=0, column=2, padx=10, pady=10)

info_button = tk.Button(root, text="About", command=show_info)
info_button.grid(row=2, column=1, padx=10, pady=10)

run_button = tk.Button(root, text="Run", command=run_script, state=tk.DISABLED)
run_button.grid(row=1, column=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
