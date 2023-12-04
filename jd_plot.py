import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import beta, triang


def calculate_cdf_triangular(min_val, max_val, likely_val, x):
    """
    Calculate the CDF for triangular distribution at point x.
    """
    c = (likely_val - min_val) / (max_val - min_val)
    return triang.cdf(x, c, loc=min_val, scale=max_val - min_val)


# Sample data
min_val, max_val, likely_val = 40000, 1200000, 250000  # 'Value' data
min_index, max_index, likely_index = 3, 8, 5  # 'Soil Quality' data

# Generate a range of values for 'Value' and 'Soil Quality'
value_range = np.linspace(min_val, max_val, 100)
index_range = np.linspace(min_index, max_index, 100)

# Generating CDF values for 'Value' and 'Soil Quality'
value_cdf = [calculate_cdf_triangular(min_val, max_val, likely_val, v) for v in value_range]
index_cdf = [calculate_cdf_triangular(min_index, max_index, likely_index, i) for i in index_range]

# Create meshgrid for 3D plot
X, Y = np.meshgrid(value_range, index_range)
Z = np.outer(value_cdf, index_cdf)

# Plotting the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Value')
ax.set_ylabel('Soil Quality')
ax.set_zlabel('Joint CDF')

plt.title('Joint Distribution of Value and Soil Quality')
plt.show()
