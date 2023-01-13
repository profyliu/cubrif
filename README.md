# Description

Build random forests for large data sets using CUDA. 
This is the GPU-enabled version of [brif](URL 'https://pypi.org/project/brif/'). 
The same program is available on [CRAN](URL 'https://cran.r-project.org/web/packages/brif/index.html') for R users. 

# Build from source

## Prerequisites

An Nvidia graphics / compute card must be present and the [CUDA Toolkit](URL 'https://developer.nvidia.com/cuda-toolkit') must be installed. 

For Windows, Microsoft Visual Studio [Build Tools for C++](URL 'https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild?view=vs-2022') must be installed. For Linux and MacOS, some C++ build tool chain (e.g., gcc) is required.  

Python [build](URL 'https://pypa-build.readthedocs.io/en/stable/') is required, can be installed via

```bash
pip install build
```

The pandas and numpy packages are required, can be installed via

```bash
pip install pandas numpy
```

## Build and install on Windows

Clone (or download as zip and extract) this project to a local directory. 

Search in the Windows search bar and run as administrator the "x64 Native Tools Command Prompt for VS 2022". 
In the command window thus opened, cd into the project root directory, and run

```bash
mkdir build
cd build
cmake ../
```

If successful, the file cubrif.sln (among other files) will be generated, then run

```bash
MSBuild.exe cubrif.sln /p:Configuration=Release
```

If successful, several files will be created in the Release subfolder. Important ones include cubrif.lib, cubrif.dll and cubrif_main.exe. cubrif.lib will be used in building python package, cubrif.dll will be used in runtime, and cubrif_main.exe is a standalone executable.  

Copy cubrif.lib to the project root directory:

```bash
copy Release\cubrif.lib ..\
```

Now go back to the project root and build the Python package, as follows

```bash
cd ..
python -m build
```

If successful, the package, e.g., cubrif-1.4.0.tar.gz, will be create in the dist subfolder. 

Install the package by

```bash
pip install dist/cubrif-1.4.0.tar.gz
```

To use the package, the cubrif.dll must be visible to python, for example:

```python
import os
os.add_dll_directory("C:/path/to/project/build/Release")
from cubrif import cubrif
```

## Build and install on Ubuntu

The build process is similar, but use 'make' instead of MSBuild.exe, and the dynamic library file generated will be libcubrif.so instead of cubrif.dll.

```bash
mkdir build
cd build
cmake ../
make
cp libcubrif.so ../
cd ..
python3 -m build
pip install dist/cubrif-1.4.0.tar.gz
```

In the above step, if "python3 -m build" does not work, use the equivalent command 
```bash
python3 setup.py sdist bdist_wheel
```

To use the package, the *libcubrif.so* must be visible to python. Either copy libcubrif.so to usr/lib or use os.add_dll_directory() as described above. For example,

```bash
sudo cp libcubrif.so /usr/lib
```

or in python,

```python
import os
os.add_dll_directory("C:/path/to/project/build/Release")
```


# Usage Examples

```python
from cubrif import cubrif
import pandas as pd

# Create a brif object with default parameters.
bf = cubrif.cubrif()  

# Display the current parameter values. 
bf.get_param()  

# To change certain parameter values, e.g.:
bf.set_param({'ntrees':10, 'nthreads':2, 'GPU':1})  

# Or simply:
bf.ntrees = 50

# Load input data frame. Data must be a pandas data frame with appropriate headers.
df = pd.read_csv("auto.csv")

# Train the model
bf.fit(df, 'origin')  # specify the target column name

# Or equivalently
bf.fit(df, 7)  # specify the target column index

# Make predictions 
# The target variable column must be excluded, and all other columns should appear in the same order as in training
# Here, predict the first 10 rows of df
pred_labels = bf.predict(df.iloc[0:10, 0:7], type='class')  # return a list containing the predicted class labels
pred_scores = bf.predict(df.iloc[0:10, 0:7], type='score')  # return a data frame containing predicted probabilities by class

# Note: for a regression problem (i.e., when the response variable is numeric type), the predict function will always return a list containing the predicted values

```

# Parameters
**tmp_preddata**
a character string specifying a filename to save the temporary scoring data. Default is "tmp_brif_preddata.txt".

**n_numeric_cuts**	
an integer value indicating the maximum number of split points to generate for each numeric variable.

**n_integer_cuts**	
an integer value indicating the maximum number of split points to generate for each integer variable.

**max_integer_classes**
an integer value. If the target variable is integer and has more than max_integer_classes unique values in the training data, then the target variable will be grouped into max_integer_classes bins. If the target variable is numeric, then the smaller of max_integer_classes and the number of unique values number of bins will be created on the target variables and the regression problem will be solved as a classification problem.

**max_depth**
an integer specifying the maximum depth of each tree. Maximum is 40.

**min_node_size**	
an integer specifying the minimum number of training cases a leaf node must contain.

**ntrees**
an integer specifying the number of trees in the forest.

**ps**
an integer indicating the number of predictors to sample at each node split. Default is 0, meaning to use sqrt(p), where p is the number of predictors in the input.

**max_factor_levels**
an integer. If any factor variables has more than max_factor_levels, the program stops and prompts the user to increase the value of this parameter if the too-many-level factor is indeed intended.

**seed**
a positive integer, random number generator seed.

**nthreads**
an integer specifying the number of threads used by the program. This parameter takes effect only on systems supporting OpenMP.

**blocksize**
an integer specifying the CUDA thread block size. Must be a multiple of 64, and no more than 1024. 

**GPU**
an integer (0, 1 or 2). 0: Do not use the GPU (for small datasets, e.g., less than 100,000 rows, using GPU is slower). 1: Force use the GPU. 2: Use GPU to evalute splits only when the node size is greater than or equal to n_lb_GPU. 

**n_lb_GPU**
an integer specifying the threshold number of rows in the training data to use GPU for training. This parameter takes effect only when GPU = 2. 

**vote_method**
an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count and an average is taken on the sum over all leaves; 1: each leaf contributes an intra-node fraction which is then averaged over all leaves with equal weight.

**na_numeric**
a numeric value, substitute for 'nan' in numeric variables.

**na_integer**
an integer value, substitute for 'nan' in integer variables.

**na_factor**
a character string, substitute for missing values in factor variables. 

**type**
a character string indicating the return content of the predict function. For a classification problem, "score" means the by-class probabilities and "class" means the class labels (i.e., the target variable levels). For regression, the predicted values are returned. This is a parameter for the predict function, not an attribute of the brif object. 

