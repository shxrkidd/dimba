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

## Data Providers

This project uses the following free football data sources:

- [StatsBomb](https://statsbomb.com/)
- [FBref](https://fbref.com/)
- [Club Elo](https://clubelo.com/)
- [FiveThirtyEight](https://projects.fivethirtyeight.com/soccer-predictions/)
- [Understat](https://understat.com/)

These sources provide a wide range of football statistics and predictions, which are used for model training and evaluation in this project.

## Commercial Data Providers

If you would like to use paid or commercial data providers, this project can be adapted to work with APIs such as:

- [Second Spectrum](https://www.secondspectrum.com/)
- [StatsPerform (Opta)](https://www.statsperform.com/)
- [Metrica Sports](https://metrica-sports.com/)
- [Signality](https://signality.com/)
- [Sportradar](https://sportradar.com/)
- [Bet365](https://www.bet365.com/) / [Betfair](https://www.betfair.com/) APIs

These providers offer advanced, real-time, and proprietary football data for professional and commercial use.
