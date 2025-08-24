# Model Evaluation Script

## Overview
`evaluate_model.py` is a standalone script that evaluates the trained coffee price prediction model and analyzes prediction accuracy, particularly focusing on identifying dates with large prediction errors.

## Features

- **Loads pre-trained model**: Uses the saved model from training without needing to retrain
- **Delta analysis**: Calculates differences between predicted and actual values
- **Date mapping**: Maps predictions back to actual dates for temporal analysis
- **Huge difference detection**: Identifies dates with unusually large prediction errors
- **Comprehensive metrics**: Provides MSE, MAE, RMSE, R², and MAPE
- **Sorted results**: Orders predictions by absolute delta in descending order
- **CSV export**: Saves detailed results for further analysis

## Configuration

All settings are configured via constants at the top of the script:

```python
# Configuration Constants
MODEL_PATH = './best_coffee_model'          # Path to trained model
DATA_PATH = 'data/test - Sheet1.csv'        # Path to data file
USE_TOY_DATA = False                        # Use subset for testing
THRESHOLD_PERCENTILE = 90                   # Percentile for "huge differences"
TOP_N_PREDICTIONS = 20                      # Number of worst predictions to show
SAVE_CSV = True                             # Save results to CSV
CONTEXT_LENGTH = 20                         # Model context length
PREDICTION_LENGTH = 1                       # Model prediction length
```

## Usage

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run evaluation (no arguments needed)
python evaluate_model.py
```

To customize behavior, edit the constants at the top of `evaluate_model.py`:
- Set `USE_TOY_DATA = True` for testing with subset
- Change `THRESHOLD_PERCENTILE` to adjust sensitivity (higher = fewer "huge" differences)
- Modify `TOP_N_PREDICTIONS` to show more/fewer worst cases
- Set `SAVE_CSV = False` to skip CSV export

## Output Sections

1. **Model Performance Summary**: Overall metrics (MSE, MAE, RMSE, R², MAPE)
2. **Delta Analysis**: Statistics about prediction errors
3. **Top N Worst Predictions**: Ranked by absolute difference
4. **Dates with Huge Differences**: Dates exceeding the threshold

## CSV Output
When using `--save_csv`, the script creates `evaluation_results.csv` with columns:
- `date`: Prediction date
- `actual`: Actual price
- `predicted`: Predicted price  
- `raw_delta`: Predicted - Actual (signed difference)
- `absolute_delta`: |Predicted - Actual|
- `percentage_error`: (Predicted - Actual) / Actual * 100
- `absolute_percentage_error`: |percentage_error|
- `is_huge_difference`: Boolean flag for threshold exceedance

## Example Output
```
================================================================================
MODEL EVALUATION SUMMARY
================================================================================
Overall Model Performance:
  MSE (Mean Squared Error): 67.06
  MAE (Mean Absolute Error): 6.07
  RMSE (Root Mean Squared Error): 8.19
  R² Score: 0.9703
  MAPE (Mean Absolute Percentage Error): 1.83%

Delta Analysis:
  Mean Absolute Delta: 6.07
  Median Absolute Delta: 4.60
  Max Absolute Delta: 29.14
  Threshold for 'Huge' Differences: 16.46 (90th percentile)
  Percentage Error Threshold: 4.97% (90th percentile)
  Total Predictions: 236
  Huge Differences: 12 (5.1%)

================================================================================
TOP 20 WORST PREDICTIONS (Largest Absolute Deltas)
================================================================================
Rank  Date         Actual     Predicted  Delta      % Error   
--------------------------------------------------------------------------------
1     2025-04-07   342.90     372.04     29.14      8.50      %
2     2024-12-02   297.70     326.75     29.05      9.76      %
...
```

## Requirements
- Trained model must exist (run `baseline_model.py` first)
- Same data format as training data
- All dependencies from `pyproject.toml` installed
