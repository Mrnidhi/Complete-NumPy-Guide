# Complete NumPy Guide

A comprehensive guide to NumPy functions and operations focused on data analysis and scientific computing workflows.


# Getting Started with NumPy Locally

To get started with NumPy locally, you can follow these steps to set up your environment and clone the repository.

## Setting Up Your Local Environment

### Step 1: Install Python

First, ensure you have Python installed on your system. You can download Python from the [official website](https://www.python.org/).

### Step 2: Fork the Repository

Fork the repository to your own GitHub account by visiting [complete-numpy-guide](https://github.com/Mrnidhi/Complete-NumPy-Guide) and clicking the "Fork" button in the top-right corner.

### Step 3: Clone the Forked Repository

Clone your forked repository to your local machine. Open a terminal or command prompt and run:

```sh
git clone https://github.com/Mrnidhi/Complete-NumPy-Guide.git
cd Complete-NumPy-Guide
```


### Step 4: Create a Virtual Environment (optional)

Creating a virtual environment is a good practice to manage dependencies for your projects. Run the following command:

```sh
python -m venv numpy_env
```

Activate the virtual environment:

- On Windows:
  ```sh
  numpy_env\Scripts\activate
  ```
- On macOS/Linux:
  ```sh
  source numpy_env/bin/activate
  ```

To deactivate the virtual environment, run:

- On Windows:
  ```sh
  numpy_env\Scripts\deactivate.bat
  ```
- On macOS/Linux:
  ```sh
  deactivate
  ```

### Step 5: Install Required Libraries

With the virtual environment activated, install NumPy and Jupyter:

```sh
pip install numpy jupyter
```

### Step 6: Open Your Code Editor

You can use your favorite code editor like Visual Studio Code or PyCharm. Open the cloned repository folder in your code editor.

### Step 7: Create a Jupyter Notebook

Create a new Jupyter Notebook file in your code editor:

- In Visual Studio Code, click on the "New File" icon or press `Ctrl+N`, then save the file with a `.ipynb` extension.
- In PyCharm, right-click on the project folder, select "New", and then "Jupyter Notebook".
- **Else**, if these options don't work or you are using an editor that doesn't support Jupyter Notebooks, run the following command in your terminal:
  ```sh
  jupyter notebook
  ```
  This will open Jupyter Notebook in your web browser.

## Using Google Colab

If you prefer not to set up things locally, you can use Google Colab, which allows you to run Python code in your browser without any setup.

Go to [Google Colab](https://colab.research.google.com/) and start a new notebook. You can start using NumPy immediately by importing it:

```python
import numpy as np
```

# Why NumPy?

NumPy forms the foundation of the Python data science ecosystem:

- **Performance**: Vectorized operations are 10-100x faster than Python loops
- **Memory efficiency**: Contiguous memory storage with precise data typing
- **Scientific capabilities**: Linear algebra, Fourier transforms, random sampling
- **Ecosystem integration**: Core library for pandas, scikit-learn, TensorFlow, etc.

## Basic Import Pattern

```python
import numpy as np
```

## Array Creation & Manipulation

### Creating Arrays

```python
# From Python lists
arr = np.array([1, 2, 3, 4, 5])                   # 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])         # 2D array

# Specialized arrays
zeros = np.zeros((3, 4))                          # Array of zeros
ones = np.ones((2, 3))                            # Array of ones
identity = np.eye(3)                              # 3x3 identity matrix
range_arr = np.arange(0, 10, 2)                   # Values from 0 to 10, step 2
linear_space = np.linspace(0, 1, 5)               # 5 evenly spaced values
random_arr = np.random.rand(3, 3)                 # Random values from [0,1)
```

### Reshaping & Transforming

```python
# Reshaping
arr = np.arange(12)
arr_reshaped = arr.reshape(3, 4)             # Reshape to 3x4 array
arr_transposed = arr_reshaped.T              # Transpose
arr_flattened = arr_reshaped.flatten()       # Flatten to 1D (returns copy)
arr_raveled = arr_reshaped.ravel()           # Flatten to 1D (returns view when possible)

# Adding/removing dimensions
expanded = np.expand_dims(arr, axis=0)       # Add new axis at position 0
squeezed = np.squeeze(expanded)              # Remove single-dimensional entries

# Stacking arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vertical = np.vstack([a, b])                 # Stack vertically [[1,2,3], [4,5,6]]
horizontal = np.hstack([a, b])               # Stack horizontally [1,2,3,4,5,6]
```

### Indexing & Slicing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Basic indexing
arr[0, 0]                # First element (1)
arr[1, -1]               # Last element of second row (8)

# Slicing [start:stop:step]
arr[:2, 1:3]             # First two rows, second and third columns
arr[::2, ::2]            # Every 2nd row and every 2nd column

# Boolean indexing
mask = arr > 5
filtered = arr[mask]     # Elements where condition is True, returns 1D array

# Fancy indexing
arr[[0, 2], [1, 3]]      # Elements at positions (0,1) and (2,3)
```

### Broadcasting

Broadcasting allows operations between arrays of different shapes:

```python
# Add scalar to array
arr = np.array([1, 2, 3])
arr + 5                         # [6, 7, 8]

# Operations between arrays of different shapes
a = np.array([[1, 2, 3]])       # Shape (1, 3)
b = np.array([[1], [2], [3]])   # Shape (3, 1)
a + b                           # Shape (3, 3) through broadcasting
```

## Statistical Operations

### Basic Statistics

```python
data = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Descriptive statistics
np.min(data)             # Minimum value
np.max(data)             # Maximum value
np.mean(data)            # Mean (average)
np.median(data)          # Median
np.std(data, ddof=1)     # Standard deviation (default ddof=0)
np.var(data, ddof=1)     # Variance
np.percentile(data, 75)  # 75th percentile

# Multi-dimensional statistics
np.mean(matrix, axis=0)  # Mean of each column
np.sum(matrix, axis=1)   # Sum of each row

# Finding extrema
np.argmin(data)          # Index of minimum value
np.argmax(data)          # Index of maximum value
```

### Data Processing

```python
# Handling missing values
data = np.array([1, 2, np.nan, 4, 5])
np.isnan(data)                   # Boolean mask for NaN values
np.nanmean(data)                 # Mean ignoring NaN values
np.nanstd(data)                  # Standard deviation ignoring NaN values

# Cumulative operations
np.cumsum(data)                  # Cumulative sum
np.cumprod(np.array([1, 2, 3]))  # Cumulative product

# Sorting
sorted_data = np.sort(data)                # Sort array
sort_indices = np.argsort(data)            # Get indices that would sort array

# Unique values
unique_values = np.unique(np.array([1, 2, 2, 3, 3, 3]))
```

## Linear Algebra

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

# Matrix operations
np.dot(a, b)              # Matrix multiplication
a @ b                     # Matrix multiplication (Python 3.5+)
np.dot(a, v)              # Matrix-vector multiplication

# Matrix decompositions
eigenvalues, eigenvectors = np.linalg.eig(a)
u, s, vh = np.linalg.svd(a)      # Singular Value Decomposition

# Solving linear systems
x = np.linalg.solve(a, v)        # Solve for x in a·x = v

# Matrix properties
np.linalg.det(a)                 # Determinant
np.trace(a)                      # Trace (sum of diagonal elements)
np.linalg.matrix_rank(a)         # Rank
np.linalg.inv(a)                 # Inverse
np.linalg.norm(a, ord='fro')     # Frobenius norm
```

## Random Number Generation

### Reproducible Results

```python
# Set seed for reproducibility
np.random.seed(42)

# NumPy 1.17+ style (preferred)
rng = np.random.default_rng(42)
```

### Sampling Distributions

```python
# Basic distributions
np.random.rand(3, 3)                        # Uniform distribution [0,1)
np.random.randn(3, 3)                       # Standard normal distribution
np.random.randint(0, 10, size=(3, 3))       # Random integers from 0 to 9

# Common statistical distributions
np.random.normal(loc=0, scale=1, size=1000)   # Normal/Gaussian
np.random.binomial(n=10, p=0.5, size=1000)    # Binomial 
np.random.poisson(lam=5, size=1000)           # Poisson

# Sampling from data
data = np.array([10, 20, 30, 40, 50])
np.random.choice(data, size=10, replace=True)                     # Sample with replacement
np.random.choice(data, size=3, replace=False, p=[0.1,0.2,0.4,0.2,0.1])  # Sample without replacement
```

## Practical Examples

### Example 1: Outlier Detection & Removal

```python
# Generate sample data with outliers
data = np.random.normal(0, 1, 1000)
data[0] = 10  # Insert an outlier

# Z-score method for outlier detection
z_scores = (data - np.mean(data)) / np.std(data)
outliers = np.abs(z_scores) > 3

# Remove outliers
clean_data = data[~outliers]

# IQR method (alternative approach)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
iqr_outliers = (data < lower_bound) | (data > upper_bound)
```

### Example 2: Feature Scaling

```python
# Sample multi-feature dataset
X = np.random.randn(100, 4) * np.array([1, 10, 100, 1000])

# Min-Max scaling [0, 1]
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_minmax = (X - X_min) / (X_max - X_min)

# Standardization (Z-score normalization)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=1)
X_standardized = (X - X_mean) / X_std
```

### Example 3: Moving Window Calculations

```python
# Generate time series data
time_series = np.cumsum(np.random.normal(0, 1, 1000))

# Rolling window functions
def rolling_window(a, window):
    """Create rolling window views into array a."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# Calculate moving average (3 different methods)
window_size = 5

# Method 1: Convolution
weights = np.ones(window_size) / window_size
moving_avg1 = np.convolve(time_series, weights, mode='valid')

# Method 2: Rolling window
windows = rolling_window(time_series, window_size)
moving_avg2 = np.mean(windows, axis=1)

# Method 3: Using pandas (if available)
# import pandas as pd
# moving_avg3 = pd.Series(time_series).rolling(window_size).mean().values[window_size-1:]
```

### Example 4: Principal Component Analysis (Simple Implementation)

```python
# Generate correlated data
n_samples = 100
t = np.linspace(0, 2 * np.pi, n_samples)
x = np.sin(t) + 0.1 * np.random.randn(n_samples)
y = np.cos(t) + 0.1 * np.random.randn(n_samples)
data = np.column_stack((x, y))

# Step 1: Center the data
data_centered = data - np.mean(data, axis=0)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(data_centered, rowvar=False)

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
eigenvalues = eigenvalues[::-1]  # Sort in descending order
eigenvectors = eigenvectors[:, ::-1]  # Reorder eigenvectors accordingly

# Step 4: Project data onto principal components
pca_data = data_centered @ eigenvectors

# Step 5: Reconstruction from first principal component only
reconstructed = pca_data[:, 0:1] @ eigenvectors[:, 0:1].T + np.mean(data, axis=0)
```

## Performance Tips

### Memory Layout

NumPy arrays are stored in contiguous memory blocks. Understanding memory layout can greatly improve performance:

```python
# C-contiguous vs Fortran-contiguous
c_array = np.zeros((1000, 1000), order='C')    # Row-major (C-style)
f_array = np.zeros((1000, 1000), order='F')    # Column-major (Fortran-style)

# Check memory layout
c_array.flags['C_CONTIGUOUS']    # True
f_array.flags['F_CONTIGUOUS']    # True

# Choose appropriate layout based on access patterns
# - Row-wise operations: use C order
# - Column-wise operations: use F order
```

### Vectorization

Always prefer vectorized operations over loops:

```python
# Slow: using loops
def slow_euclidean(x, y):
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

# Fast: vectorized
def fast_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Even faster: using built-in norm
def fastest_euclidean(x, y):
    return np.linalg.norm(x - y)
```

### Working with Large Arrays

```python
# Memory-mapped arrays for large datasets
mmap_array = np.memmap('large_data.npy', dtype='float32', mode='w+', shape=(10000, 10000))

# Work with small sections at a time
for i in range(0, 10000, 100):
    mmap_array[i:i+100] = np.random.random((100, 10000))

# Flush changes to disk and close
mmap_array.flush()
del mmap_array
```

### Views vs Copies

Understanding when NumPy returns views vs copies can prevent unexpected behavior:

```python
# Views (changes to one affect the other)
a = np.arange(10)
b = a[2:7]         # b is a view of a
b[0] = 99          # This changes a[2] as well

# Copies (independent)
a = np.arange(10)
b = a[2:7].copy()  # Explicit copy
b[0] = 99          # This doesn't affect a

# Note: Advanced indexing always returns copies
x = np.arange(10)
y = x[[0, 1, 2]]   # y is a copy, not a view
```

## Resources

- [Official NumPy Documentation](https://numpy.org/doc/stable/)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [NumPy Illustrated: The Visual Guide to NumPy](https://medium.com/better-programming/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)

## License

This project is licensed under the MIT License - see the LICENSE file for details.