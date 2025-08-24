import pandas as pd
import numpy as np
import torch
from transformers import PatchTSTForPrediction, PatchTSTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Configuration Constants
MODEL_PATH = './best_coffee_model'
DATA_PATH = 'data/test - Sheet1.csv'
USE_TOY_DATA = False
THRESHOLD_PERCENTILE = 90
TOP_N_PREDICTIONS = 20
SAVE_CSV = True
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 1

def load_and_prepare_data(data_path="data/test - Sheet1.csv", use_toy_data=False):
    """Load and prepare the coffee price data"""
    try:
        df = pd.read_csv(data_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Use 'Price' column instead of 'Close'
        df = df.rename(columns={"Price": "Close"})
        
        print(f"Full data loaded successfully. Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        if use_toy_data:
            df = df.tail(100)
            print(f"Using subset for testing. Shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found. Make sure the file is in the correct directory.")
        exit()
    
    # Ensure business day frequency
    df = df.asfreq("B", method="ffill")
    time_series = df["Close"].values
    
    return df, time_series

def create_sequences(time_series, context_length=20, prediction_length=1):
    """Create sequences for model input"""
    X, Y = [], []
    date_indices = []  # To track which dates correspond to predictions
    
    for i in range(len(time_series) - context_length - prediction_length + 1):
        X.append(time_series[i : i + context_length])
        Y.append(time_series[i + context_length : i + context_length + prediction_length])
        date_indices.append(i + context_length)  # Index of the date being predicted
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, date_indices

def make_predictions(model, X_data):
    """Generate predictions for a dataset"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_data)):
            input_tensor = torch.tensor(X_data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            outputs = model.forward(past_values=input_tensor)
            pred = outputs.prediction_outputs.squeeze().numpy()
            predictions.append(pred)
    
    return np.array(predictions)

def calculate_deltas_with_dates(predictions, actuals, date_indices, df_dates, threshold_percentile=90):
    """
    Calculate deltas between predictions and actuals, map to dates, and identify huge differences
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values  
        date_indices: Indices mapping predictions to dates
        df_dates: DatetimeIndex from original dataframe
        threshold_percentile: Percentile to define "huge" differences (default: 90th percentile)
    
    Returns:
        DataFrame with predictions, actuals, deltas, dates, and analysis
    """
    # Flatten arrays if needed
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()
    
    # Calculate various delta metrics
    absolute_delta = np.abs(predictions_flat - actuals_flat)
    raw_delta = predictions_flat - actuals_flat
    percentage_error = (raw_delta / actuals_flat) * 100
    absolute_percentage_error = np.abs(percentage_error)
    
    # Map to actual dates
    prediction_dates = [df_dates[idx] for idx in date_indices]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'date': prediction_dates,
        'actual': actuals_flat,
        'predicted': predictions_flat,
        'raw_delta': raw_delta,
        'absolute_delta': absolute_delta,
        'percentage_error': percentage_error,
        'absolute_percentage_error': absolute_percentage_error
    })
    
    # Sort by absolute delta in descending order
    results_df = results_df.sort_values('absolute_delta', ascending=False).reset_index(drop=True)
    
    # Calculate threshold for "huge" differences
    delta_threshold = np.percentile(absolute_delta, threshold_percentile)
    percentage_threshold = np.percentile(absolute_percentage_error, threshold_percentile)
    
    # Mark huge differences
    results_df['is_huge_difference'] = (
        (results_df['absolute_delta'] >= delta_threshold) | 
        (results_df['absolute_percentage_error'] >= percentage_threshold)
    )
    
    return results_df, delta_threshold, percentage_threshold

def print_evaluation_summary(results_df, delta_threshold, percentage_threshold):
    """Print comprehensive evaluation summary"""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    
    # Basic metrics
    predictions = results_df['predicted'].values
    actuals = results_df['actual'].values
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"Overall Model Performance:")
    print(f"  MSE (Mean Squared Error): {mse:.2f}")
    print(f"  MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Delta analysis
    print(f"\nDelta Analysis:")
    print(f"  Mean Absolute Delta: {results_df['absolute_delta'].mean():.2f}")
    print(f"  Median Absolute Delta: {results_df['absolute_delta'].median():.2f}")
    print(f"  Max Absolute Delta: {results_df['absolute_delta'].max():.2f}")
    print(f"  Threshold for 'Huge' Differences: {delta_threshold:.2f} (90th percentile)")
    print(f"  Percentage Error Threshold: {percentage_threshold:.2f}% (90th percentile)")
    
    # Count huge differences
    huge_differences = results_df[results_df['is_huge_difference']]
    print(f"  Total Predictions: {len(results_df)}")
    print(f"  Huge Differences: {len(huge_differences)} ({len(huge_differences)/len(results_df)*100:.1f}%)")

def print_worst_predictions(results_df, top_n=20):
    """Print the worst predictions (largest absolute deltas)"""
    print(f"\n" + "="*80)
    print(f"TOP {top_n} WORST PREDICTIONS (Largest Absolute Deltas)")
    print("="*80)
    print(f"{'Rank':<5} {'Date':<12} {'Actual':<10} {'Predicted':<10} {'Delta':<10} {'% Error':<10}")
    print("-" * 80)
    
    for i, row in results_df.head(top_n).iterrows():
        print(f"{i+1:<5} {row['date'].strftime('%Y-%m-%d'):<12} "
              f"{row['actual']:<10.2f} {row['predicted']:<10.2f} "
              f"{row['absolute_delta']:<10.2f} {row['absolute_percentage_error']:<10.2f}%")

def print_huge_differences(results_df):
    """Print all dates with huge differences"""
    huge_differences = results_df[results_df['is_huge_difference']]
    
    print(f"\n" + "="*80)
    print(f"DATES WITH HUGE DIFFERENCES ({len(huge_differences)} total)")
    print("="*80)
    
    if len(huge_differences) > 0:
        print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Delta':<10} {'% Error':<10} {'Direction':<10}")
        print("-" * 80)
        
        for _, row in huge_differences.iterrows():
            direction = "Over" if row['raw_delta'] > 0 else "Under"
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} "
                  f"{row['actual']:<10.2f} {row['predicted']:<10.2f} "
                  f"{row['absolute_delta']:<10.2f} {row['absolute_percentage_error']:<10.2f}% "
                  f"{direction:<10}")
    else:
        print("No huge differences found with current threshold.")

def save_results_to_csv(results_df, filename="evaluation_results.csv"):
    """Save detailed results to CSV"""
    results_df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to: {filename}")

def save_top_k_predictions(results_df, top_k, filename="top_k_worst_predictions.csv"):
    """Save top k worst predictions to separate CSV"""
    top_k_df = results_df.head(top_k).copy()
    top_k_df.to_csv(filename, index=False)
    print(f"Top {top_k} worst predictions saved to: {filename}")

def plot_evaluation_results(results_df, predictions, actuals):
    """Create comprehensive evaluation plots"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Predicted vs Actual scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(actuals, predictions, alpha=0.6, color='blue')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    r2 = r2_score(actuals, predictions)
    plt.title(f'Predicted vs Actual\nR² = {r2:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time series of predictions vs actuals
    plt.subplot(2, 3, 2)
    indices = range(len(actuals))
    plt.plot(indices, actuals, 'b-', label='Actual', alpha=0.7, linewidth=1)
    plt.plot(indices, predictions, 'r-', label='Predicted', alpha=0.7, linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title('Time Series: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals (prediction errors) over time
    plt.subplot(2, 3, 3)
    residuals = predictions - actuals
    plt.plot(indices, residuals, 'g-', alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title('Residuals Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of absolute errors
    plt.subplot(2, 3, 4)
    absolute_errors = np.abs(residuals)
    plt.hist(absolute_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Errors')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Top 10 worst predictions with dates
    plt.subplot(2, 3, 5)
    top_10 = results_df.head(10)
    dates_str = [d.strftime('%m-%d') for d in top_10['date']]
    y_pos = range(len(dates_str))
    plt.barh(y_pos, top_10['absolute_delta'], color='red', alpha=0.7)
    plt.yticks(y_pos, dates_str)
    plt.xlabel('Absolute Delta')
    plt.title('Top 10 Worst Predictions')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Percentage errors over time
    plt.subplot(2, 3, 6)
    percentage_errors = (residuals / actuals) * 100
    plt.plot(indices, percentage_errors, 'purple', alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Errors Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nEvaluation plots saved as: evaluation_plots.png")

def main():
    print("Loading and preparing data...")
    df, time_series = load_and_prepare_data(DATA_PATH, USE_TOY_DATA)
    
    print("Creating sequences...")
    X, Y, date_indices = create_sequences(time_series, CONTEXT_LENGTH, PREDICTION_LENGTH)
    
    # Split data (same as training split)
    X_train, X_eval, Y_train, Y_eval = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Get corresponding date indices for eval set
    train_size = len(X_train)
    eval_date_indices = date_indices[train_size:]
    
    print(f"Evaluation set size: {len(X_eval)} predictions")
    print(f"Date range for evaluation: {df.index[eval_date_indices[0]]} to {df.index[eval_date_indices[-1]]}")
    
    # Load the trained model
    print(f"Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path '{MODEL_PATH}' not found!")
        print("Make sure you have trained a model first using baseline_model.py")
        exit()
    
    model = PatchTSTForPrediction.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = make_predictions(model, X_eval)
    
    # Calculate deltas and map to dates
    print("Calculating deltas and mapping to dates...")
    results_df, delta_threshold, percentage_threshold = calculate_deltas_with_dates(
        predictions, Y_eval, eval_date_indices, df.index, THRESHOLD_PERCENTILE
    )
    
    # Print comprehensive analysis
    print_evaluation_summary(results_df, delta_threshold, percentage_threshold)
    print_worst_predictions(results_df, TOP_N_PREDICTIONS)
    print_huge_differences(results_df)
    
    # Save results if requested
    if SAVE_CSV:
        save_results_to_csv(results_df)
        save_top_k_predictions(results_df, TOP_N_PREDICTIONS)
    
    # Create evaluation plots
    print("\nCreating evaluation plots...")
    predictions_flat = predictions.flatten()
    actuals_flat = Y_eval.flatten()
    plot_evaluation_results(results_df, predictions_flat, actuals_flat)
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
