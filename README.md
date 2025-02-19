# Binary Cascade Analyzer

## Description
The Binary Cascade Analyzer is a Python tool designed to detect and analyze patterns in binary strings. It uses a cascade approach to identify repeating sequences and can work with both randomly generated and artificially created data. Note: This is unfinished, still a work in progress and it might have errores (28/07/2024), but any suggestions or errors notification, please send me an email (k i n e t i c n e x u s @ g m a i l.com)

## Features
- Generate binary strings with customizable value ranges
- Create and analyze cascades from binary strings
- Detect substantial differences in patterns using statistical methods
- Perform experiments on both random and artificial data
- Calculate various statistical measures (mean, median, standard deviation)
- Conduct statistical tests (Shapiro-Wilk, Anderson-Darling, etc.)
- Visualize results with histograms

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## Installation
Clone this repository and install the required packages:

## Usage
Run the main script to perform analysis on random and artificial data:
(use artificial data and hide known patterns to understand if the script is detecting them!)

```bash
python binary_cascade_analyzer.py
```

## Main Components
1. `BinaryCascadeAnalyzer`: The core class for generating and analyzing binary cascades.
2. `calculate_statistics`: Computes statistical measures on the results.
3. `plot_histogram`: Visualizes the distribution of detected window sizes.
4. `statistical_checks`: Performs various statistical tests on the results.
5. `main_random`: Analyzes randomly generated data.
6. `main_artificial_data`: Analyzes artificially created data with known patterns.

## Example Output
The script provides detailed output including:
- Detected window sizes
- Statistical measures (average, median, standard deviation)
- Results of statistical tests
- Histograms of detected window sizes

## Contact
For questions or feedback, please open an issue in this repository.
