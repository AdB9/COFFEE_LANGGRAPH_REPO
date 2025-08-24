import pandas as pd
import numpy as np
import torch
from transformers import (
    PatchTSTForPrediction,
    PatchTSTConfig,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

USE_TOY_DATA = False

# --- 1. Data Preparation ---

# Define the context and prediction lengths
context_length = 20  # Use the last 20 business days of data (4 weeks)
prediction_length = 1  # Predict the next 1 day
# Load your historical coffee data
try:
    df = pd.read_csv("data/test - Sheet1.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # Sort by date to ensure chronological order (data is in reverse order)
    df = df.sort_index()

    # Use 'Price' column instead of 'Close' (adapting to the actual data structure)
    df = df.rename(columns={"Price": "Close"})

    print(f"Full data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Data columns: {df.columns.tolist()}")

    if USE_TOY_DATA:
        # USE SMALL SUBSET FOR TESTING - Take only last 100 days
        df = df.tail(100)
    print(f"Using subset for testing. Shape: {df.shape}")
    print(f"Subset date range: {df.index.min()} to {df.index.max()}")

except FileNotFoundError:
    print(
        "Error: 'data/test - Sheet1.csv' not found. Make sure the file is in the correct directory."
    )
    exit()

# Ensure business day frequency (skips weekends automatically)
df = df.asfreq("B", method="ffill")
time_series = df["Close"].values

print(f"Time series length after resampling: {len(time_series)}")
print(f"First 10 values: {time_series[:10]}")
print(f"Last 10 values: {time_series[-10:]}")

# Create sequences of data for training
X, Y = [], []
for i in range(len(time_series) - context_length - prediction_length + 1):
    X.append(time_series[i : i + context_length])
    Y.append(time_series[i + context_length : i + context_length + prediction_length])

# Convert to NumPy arrays with proper shape for PatchTST
X = np.array(X)  # Shape: (n_samples, context_length)
Y = np.array(Y)  # Shape: (n_samples, prediction_length)

print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Split data into training and evaluation sets (80% train, 20% eval)
X_train, X_eval, Y_train, Y_eval = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training set - X: {X_train.shape}, Y: {Y_train.shape}")
print(f"Evaluation set - X: {X_eval.shape}, Y: {Y_eval.shape}")

# Custom dataset class for PatchTST
class TimeSeriesDataset:
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            "past_values": self.X[idx].unsqueeze(-1),  # Add channel dimension
            "future_values": self.Y[idx].unsqueeze(-1)  # Add channel dimension
        }

# Create custom datasets
train_dataset = TimeSeriesDataset(X_train, Y_train)
eval_dataset = TimeSeriesDataset(X_eval, Y_eval)

# --- 2. Model Configuration and Loading ---

# Create a model configuration for our specific task (SMALL MODEL FOR TESTING)
config = PatchTSTConfig(
    context_length=context_length,
    prediction_length=prediction_length,
    num_input_channels=1,
    patch_length=1,  # For daily data, patch length of 1 is reasonable
    stride=1,
    hidden_size=32,  # Very small model for testing
    num_hidden_layers=2,  # Reduced layers
    num_attention_heads=2,  # Reduced heads
    intermediate_size=64,  # Reduced size
    dropout=0.1,
    attention_dropout=0.1,
    random_mask_ratio=0.1,
)

# Initialize the model with our configuration
model = PatchTSTForPrediction(config)

print("Model initialized successfully")

# --- 3. Fine-Tuning the Model ---

# Define training arguments (REDUCED FOR TESTING)
training_args = TrainingArguments(
    output_dir="./patchtst-coffee-finetuned",
    per_device_train_batch_size=8,  # Small batch size for testing
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Just 3 epochs for testing
    learning_rate=0.001,  # Higher learning rate for faster convergence
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,  # Disable since eval_loss not available
    logging_steps=5,
    save_total_limit=1,
    warmup_steps=10,  # Reduced warmup
    weight_decay=0.01,
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    report_to="none",  # Disable wandb/tensorboard
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("--- Starting Training ---")
try:
    trainer.train()
    print("--- Training Complete ---")
except Exception as e:
    print(f"Training failed with error: {e}")
    print("This might be due to insufficient data or model configuration issues.")
    exit()

# Save the best model
fine_tuned_model_path = "./best_coffee_model"
trainer.save_model(fine_tuned_model_path)
print(f"Best model saved to {fine_tuned_model_path}")


# --- 4. Making a Prediction with the Fine-Tuned Model ---

print("\n--- Making a Prediction with the Fine-Tuned Model ---")

# Load our custom fine-tuned model
loaded_model = PatchTSTForPrediction.from_pretrained(fine_tuned_model_path)

# Get the last 7 days of data from the original time series
last_week_data = time_series[-context_length:]
print(f"Input for prediction (last 7 days): \n{last_week_data}")

# Prepare the input tensor for the model
input_tensor = torch.tensor(last_week_data, dtype=torch.float32).reshape(
    1, context_length, 1
)

# Make the prediction
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model.forward(past_values=input_tensor)
    prediction = outputs.prediction_outputs.squeeze().numpy()

# Get yesterday's actual price for comparison
last_actual_price = time_series[-1]

print(f"\nLast actual closing price: {last_actual_price:.2f}")
print(f"Predicted price for the next day: {prediction:.2f}")
print(
    f"Prediction change: {((prediction - last_actual_price) / last_actual_price * 100):.2f}%"
)

# Additional analysis
print(f"\nData summary:")
print(f"Mean price: {np.mean(time_series):.2f}")
print(f"Std price: {np.std(time_series):.2f}")
print(f"Min price: {np.min(time_series):.2f}")
print(f"Max price: {np.max(time_series):.2f}")

# --- 5. Model Evaluation with Metrics and Visualization ---

print("\n" + "="*50)
print("MODEL EVALUATION: PREDICTIONS VS GROUND TRUTH")
print("="*50)

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

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate and print metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MSE (Mean Squared Error): {mse:.2f}")
    print(f"  MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# Generate predictions for training and test sets
print("Generating predictions for training set...")
train_predictions = make_predictions(loaded_model, X_train)

print("Generating predictions for test set...")
test_predictions = make_predictions(loaded_model, X_eval)

# Flatten the target arrays for metric calculation
Y_train_flat = Y_train.flatten()
Y_eval_flat = Y_eval.flatten()

# Calculate metrics
train_metrics = calculate_metrics(Y_train_flat, train_predictions, "TRAINING SET")
test_metrics = calculate_metrics(Y_eval_flat, test_predictions, "TEST SET")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Training Set Predictions vs Actual
plt.subplot(2, 2, 1)
plt.scatter(Y_train_flat, train_predictions, alpha=0.6, color='blue')
plt.plot([Y_train_flat.min(), Y_train_flat.max()], [Y_train_flat.min(), Y_train_flat.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Training Set: Predicted vs Actual\nR² = {train_metrics["R2"]:.4f}')
plt.grid(True, alpha=0.3)

# Plot 2: Test Set Predictions vs Actual
plt.subplot(2, 2, 2)
plt.scatter(Y_eval_flat, test_predictions, alpha=0.6, color='green')
plt.plot([Y_eval_flat.min(), Y_eval_flat.max()], [Y_eval_flat.min(), Y_eval_flat.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Test Set: Predicted vs Actual\nR² = {test_metrics["R2"]:.4f}')
plt.grid(True, alpha=0.3)

# Plot 3: Time Series of Training Predictions
plt.subplot(2, 2, 3)
train_indices = range(len(Y_train_flat))
plt.plot(train_indices, Y_train_flat, 'b-', label='Actual', alpha=0.7)
plt.plot(train_indices, train_predictions, 'r-', label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Training Set: Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Time Series of Test Predictions
plt.subplot(2, 2, 4)
test_indices = range(len(Y_eval_flat))
plt.plot(test_indices, Y_eval_flat, 'b-', label='Actual', alpha=0.7)
plt.plot(test_indices, test_predictions, 'r-', label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Test Set: Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary comparison
print("\n" + "="*50)
print("SUMMARY COMPARISON")
print("="*50)
print(f"{'Metric':<10} {'Training':<12} {'Test':<12} {'Difference':<12}")
print("-" * 50)
for metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:
    train_val = train_metrics[metric]
    test_val = test_metrics[metric]
    diff = test_val - train_val
    print(f"{metric:<10} {train_val:<12.2f} {test_val:<12.2f} {diff:<12.2f}")

print(f"{'R2':<10} {train_metrics['R2']:<12.4f} {test_metrics['R2']:<12.4f} {test_metrics['R2'] - train_metrics['R2']:<12.4f}")

# Model performance assessment
print("\n" + "="*50)
print("MODEL PERFORMANCE ASSESSMENT")
print("="*50)

if test_metrics['R2'] > 0.7:
    performance = "EXCELLENT"
elif test_metrics['R2'] > 0.5:
    performance = "GOOD"
elif test_metrics['R2'] > 0.3:
    performance = "MODERATE"
else:
    performance = "POOR"

print(f"Overall Model Performance: {performance}")
print(f"Test Set R² Score: {test_metrics['R2']:.4f}")
print(f"Test Set MAPE: {test_metrics['MAPE']:.2f}%")

if abs(train_metrics['R2'] - test_metrics['R2']) < 0.1:
    print("✅ Model shows good generalization (low overfitting)")
else:
    print("⚠️  Model may be overfitting (large gap between train/test performance)")

print(f"\nVisualization saved as 'model_evaluation.png'")
print("="*50)
