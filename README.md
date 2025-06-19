# Project Setup Guide

To run this project, you need to have Python installed along with the following Python libraries:

- `pandas`
- `tensorflow`
- `colorama`

## Installation Instructions

1. **Install Python**

   Make sure you have Python 3.7 or newer installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **(Optional) Create a Virtual Environment**

   It is recommended to use a virtual environment to manage dependencies:

   ```
   python -m venv venv
   ```

   Activate the virtual environment:

   - **Windows:**
     ```
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```
     source venv/bin/activate
     ```

3. **Install Required Libraries**

   Use `pip` to install the required packages:

   ```
   pip install pandas tensorflow colorama
   ```

4. **Verify Installation**

   You can check if the libraries are installed by running:

   ```python
   python -c "import pandas; import tensorflow; import colorama; print('All libraries are installed!')"
   ```

## Troubleshooting

- If you encounter `ModuleNotFoundError` for any of the libraries, make sure you have activated your virtual environment (if using one) and installed the packages using `pip`.
- If you have multiple versions of Python installed, use `python3` and `pip3` instead of `python` and `pip`.

## Additional Notes

- For more information on each library, visit their official documentation:
  - [pandas](https://pandas.pydata.org/)
  - [tensorflow](https://www.tensorflow.org/)
  - [colorama](https://pypi.org/project/colorama/)

## Python Version Compatibility

> **Note:** TensorFlow does **not** currently support Python 3.13 or above.  
> If you need to use TensorFlow, please use Python 3.7–3.11.  
> For Python 3.13, consider using alternative machine learning libraries such as **PyTorch** or **scikit-learn**.

## JAX Setup

This project uses [JAX](https://github.com/google/jax) for machine learning.

Install with:
```
pip install jax jaxlib flax optax
```
