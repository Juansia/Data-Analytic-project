import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import base64
from io import BytesIO
import glob

def debug_directory_structure():
    """Helper function to debug what files are actually present"""
    print("\n=== DEBUGGING DIRECTORY STRUCTURE ===")
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # List all directories in current path
    print("\nDirectories in current path:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  - {item}/")
    
    # Check for Sugar directory and its contents
    if os.path.exists('Sugar'):
        print("\nContents of Sugar directory:")
        for root, dirs, files in os.walk('Sugar'):
            level = root.replace('Sugar', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
    else:
        print("\nSugar directory not found!")
    
    # Check for Brent directory and its contents
    if os.path.exists('Brent'):
        print("\nContents of Brent directory:")
        for root, dirs, files in os.walk('Brent'):
            level = root.replace('Brent', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
    else:
        print("\nBrent directory not found!")
    
    print("=== END DEBUG ===\n")

def run_prediction(days_ahead=3, commodity='brent'):
    """
    Run price prediction for the specified commodity and number of days ahead
    
    Args:
        days_ahead: Number of days to predict (default: 3)
        commodity: Commodity to predict ('brent' or 'sugar')
        
    Returns:
        Dictionary containing prediction results and visualization
    """
    # Debug directory structure (comment out after debugging)
    debug_directory_structure()
    
    # Find the most appropriate model folder based on commodity
    folder_path = find_best_model_folder(commodity)
    print(f"Using model from: {folder_path}")
    
    # Load prediction data
    predictions, true_values = load_prediction_data(folder_path, commodity)
    
    # Get the latest date and historical data from the dataset
    last_date, historical_data = load_historical_data(commodity)
    
    # Create future dates and get future predictions
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
    future_prices = get_future_prices(predictions, days_ahead, commodity)
    
    # Get the proper price column name based on commodity
    if commodity == 'brent':
        predicted_column_name = 'Predicted_BRENT_Price'
    else:
        predicted_column_name = 'Predicted_SUGAR_Price'
    
    # Create results dataframe
    results = pd.DataFrame({
        'Day': [f"day {i+1}" for i in range(days_ahead)],
        'Date': future_dates_str,
        predicted_column_name: future_prices
    })
    
    # Generate visualization
    plot_data = generate_prediction_plot(
        last_date, future_dates, future_prices, 
        historical_data, predictions, true_values, days_ahead, folder_path, commodity
    )
    
    # Calculate statistics
    stats = calculate_statistics(historical_data, future_prices, commodity)
    
    # Prepare past predictions data
    past_predictions = prepare_past_predictions(
        last_date, predictions, true_values, days_ahead, commodity
    )
    
    # Add trend direction text
    price_direction = "INCREASE" if stats["price_change_percent"] >= 0 else "DECREASE"
    stats["price_direction"] = price_direction
    
    # Return the results with trend data for each period
    result = {
        "commodity": commodity,
        "days_ahead": days_ahead,
        "predictions": results.to_dict(orient='records'),
        "table": results.to_dict(orient='records'),  # For backward compatibility
        "statistics": stats,
        "stats": stats,  # For backward compatibility
        "historical": plot_data["historical_data"],
        "past_predictions": past_predictions,
        "plot": plot_data["plot_base64"],
        "model": os.path.basename(folder_path),
        # Add trend data for the mini charts
        "trend": future_prices.tolist()
    }
    
    # Add debug info
    print(f"Returning prediction for {commodity} with {days_ahead} days")
    print(f"Future prices (trend): {future_prices.tolist()}")
    print(f"Statistics: last_known={stats['last_known_price']:.2f}, avg_forecast={stats['avg_forecasted_price']:.2f}")
    
    return result


def find_best_model_folder(commodity='brent'):
    """Find the most appropriate model folder with prediction results
    
    Using the latest Ensemble models:
    - Brent: 1_Ensemble2025_05_27__17_10_20_Weighted
    - Sugar: 1_Ensemble2025_05_27__20_03_21_Weighted
    """
    print(f"Looking for {commodity} model folder...")
    
    # Define specific paths for each commodity using the NEW exact paths provided
    if commodity == 'brent':
        # Use the NEW exact path provided for Brent - Ensemble model
        specific_path = r"Brent\results\1_Ensemble2025_05_27__17_10_20_Weighted"
        
        print(f"Checking Brent path: {specific_path}")
        print(f"Path exists: {os.path.exists(specific_path)}")
        print(f"Preds file exists: {os.path.exists(os.path.join(specific_path, 'preds.npy'))}")
        
        # Check if path exists and contains preds.npy
        if os.path.exists(specific_path) and os.path.exists(os.path.join(specific_path, "preds.npy")):
            return specific_path
        
        # Try alternate slash style
        specific_path = "Brent/results/1_Ensemble2025_05_27__17_10_20_Weighted"
        print(f"Checking alternate Brent path: {specific_path}")
        if os.path.exists(specific_path) and os.path.exists(os.path.join(specific_path, "preds.npy")):
            return specific_path
            
    elif commodity == 'sugar':
        # Use the NEW exact path provided for Sugar - Ensemble model
        specific_path = r"Sugar\results\1_Ensemble2025_05_27__20_03_21_Weighted"
        
        print(f"Checking Sugar path: {specific_path}")
        print(f"Path exists: {os.path.exists(specific_path)}")
        print(f"Preds file exists: {os.path.exists(os.path.join(specific_path, 'preds.npy'))}")
        
        # Check if path exists and contains preds.npy
        if os.path.exists(specific_path) and os.path.exists(os.path.join(specific_path, "preds.npy")):
            return specific_path
        
        # Try alternate slash style
        specific_path = "Sugar/results/1_Ensemble2025_05_27__20_03_21_Weighted"
        print(f"Checking alternate Sugar path: {specific_path}")
        if os.path.exists(specific_path) and os.path.exists(os.path.join(specific_path, "preds.npy")):
            return specific_path
    
    # If specific paths don't work, raise an error with helpful message
    if commodity == 'brent':
        expected_path = r"Brent\results\1_Ensemble2025_05_27__17_10_20_Weighted"
    else:
        expected_path = r"Sugar\results\1_Ensemble2025_05_27__20_03_21_Weighted"
    
    # Try to find the parent directory that contains Brent/Sugar folders
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we need to go up directories to find the data
    possible_base_paths = [
        ".",  # current directory
        "..",  # parent directory
        "../..",  # grandparent directory
        "../../..",  # great-grandparent directory
        os.path.dirname(os.path.abspath(__file__))  # directory of this script
    ]
    
    print("Searching for Sugar/Brent directories in parent paths...")
    for base_path in possible_base_paths:
        test_path = os.path.join(base_path, expected_path.replace('\\', os.sep))
        print(f"Trying: {test_path}")
        if os.path.exists(test_path) and os.path.exists(os.path.join(test_path, "preds.npy")):
            return test_path
    
    raise Exception(f"Could not find model folder for {commodity}. Please ensure the following path exists: {expected_path}")


def load_prediction_data(folder_path, commodity='brent'):
    """Load prediction and true values from model output files"""
    preds_file = os.path.join(folder_path, "preds.npy")
    vals_file = os.path.join(folder_path, "vals.npy")
    
    # Check if prediction files exist
    if not os.path.exists(preds_file):
        raise Exception(f"Prediction file not found: {preds_file}")
    
    if not os.path.exists(vals_file):
        raise Exception(f"True values file not found: {vals_file}")
    
    # Load the prediction and true values
    try:
        predictions = np.load(preds_file)
        true_values = np.load(vals_file)
        
        print(f"Loaded predictions shape: {predictions.shape}")
        print(f"Loaded true values shape: {true_values.shape}")
        
        # Handle multi-dimensional predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]
            true_values = true_values[:, 0]
        
        # Flatten if necessary
        predictions = predictions.flatten()
        true_values = true_values.flatten()
        
        # Handle NaN values
        predictions = replace_nan_values(predictions, true_values, commodity)
        true_values = replace_nan_values(true_values, commodity=commodity)
        
        return predictions, true_values
        
    except Exception as e:
        raise Exception(f"Error loading prediction files from {folder_path}: {str(e)}")


def replace_nan_values(array, reference_array=None, commodity='brent'):
    """Replace NaN values in an array with a reasonable substitute"""
    if not np.isnan(array).any():
        return array
    
    # Find the last valid value
    valid_indices = ~np.isnan(array)
    if np.any(valid_indices):
        last_valid_value = array[valid_indices][-1]
    elif reference_array is not None and len(reference_array) > 0 and not np.isnan(reference_array).all():
        last_valid_value = np.nanmean(reference_array)
    else:
        last_valid_value = 80.0 if commodity == 'brent' else 65.0  # Default value based on commodity
    
    return np.where(np.isnan(array), last_valid_value, array)


def load_historical_data(commodity='brent'):
    """Load historical price data and return the latest date"""
    print(f"Loading historical data for {commodity}...")
    
    # Define data file paths based on commodity - using exact paths provided
    if commodity == 'brent':
        data_file = r"Brent\dataset\processed_data_best_corr_sentiment.csv"
        price_column = 'BRENT Close'
        date_column = 'date'
    elif commodity == 'sugar':
        data_file = r"Sugar\dataset\merged_financial_data_final.csv"
        price_column = 'SUGAR'  # Changed from 'price' to 'SUGAR'
        date_column = 'date'
    else:
        raise Exception(f"Unknown commodity: {commodity}")
    
    print(f"Looking for data file: {data_file}")
    
    # Try both backslash and forward slash versions
    alternate_paths = [
        data_file,
        data_file.replace('\\', '/'),
        "./" + data_file,
        "./" + data_file.replace('\\', '/')
    ]
    
    actual_path = None
    for path in alternate_paths:
        print(f"Checking path: {path} - Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        # Try searching in parent directories
        print("Trying parent directories...")
        possible_base_paths = [
            ".",  # current directory
            "..",  # parent directory
            "../..",  # grandparent directory
            "../../..",  # great-grandparent directory
            os.path.dirname(os.path.abspath(__file__))  # directory of this script
        ]
        
        for base_path in possible_base_paths:
            test_path = os.path.join(base_path, data_file.replace('\\', os.sep))
            print(f"Trying parent path: {test_path} - Exists: {os.path.exists(test_path)}")
            if os.path.exists(test_path):
                actual_path = test_path
                break
        
        if actual_path is None:
            raise Exception(f"Could not find historical data file for {commodity}. Tried paths: {', '.join(alternate_paths)}")
    
    try:
        df = pd.read_csv(actual_path)
        print(f"Loaded historical data from: {actual_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if date column exists
        if date_column not in df.columns:
            # Try to find a date-like column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_column = date_cols[0]
                print(f"Using date column: {date_column}")
            else:
                raise Exception(f"No date column found in {actual_path}")
        
        # Check if price column exists
        if price_column not in df.columns:
            # For sugar, try to find SUGAR column or other price-like columns
            if commodity == 'sugar':
                price_cols = [col for col in df.columns if 'sugar' in col.lower() or 'price' in col.lower()]
                if price_cols:
                    price_column = price_cols[0]
                    print(f"Using price column: {price_column}")
                else:
                    # Last resort - check for exact column name
                    if 'SUGAR' in df.columns:
                        price_column = 'SUGAR'
                    else:
                        raise Exception(f"No price column found in {actual_path}. Available columns: {df.columns.tolist()}")
            else:
                raise Exception(f"Price column '{price_column}' not found in {actual_path}. Available columns: {df.columns.tolist()}")
        
        # Convert date to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date to ensure we get the latest
        df = df.sort_values(by=date_column)
        
        # Get the last date in the dataset
        last_date = df[date_column].max()
        print(f"Last date in dataset: {last_date}")
        
        # Get historical prices for the last 30 days
        num_hist_days = min(30, len(df))
        dates = df[date_column].values[-num_hist_days:]
        prices = df[price_column].values[-num_hist_days:]
        
        # Handle NaN values in prices
        if np.isnan(prices).any():
            prices = replace_nan_values(prices, commodity=commodity)
        
        # Create historical data list
        historical_data = []
        for i in range(len(dates)):
            date_str = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            historical_data.append({
                'date': date_str,
                'price': float(prices[i])
            })
        
        return last_date, {
            'df': df,
            'dates': dates,
            'prices': prices,
            'data': historical_data,
            'price_column': price_column
        }
        
    except Exception as e:
        raise Exception(f"Error loading historical data from {actual_path}: {str(e)}")


def get_future_prices(predictions, days_ahead, commodity='brent'):
    """Extract future prices from predictions array"""
    print(f"Getting future prices for {days_ahead} days from {len(predictions)} predictions")
    
    if len(predictions) >= days_ahead:
        # Take the last 'days_ahead' predictions
        future_prices = predictions[-days_ahead:]
    else:
        # If not enough predictions, use what we have and repeat the last one
        if len(predictions) > 0:
            future_prices = list(predictions) + [predictions[-1]] * (days_ahead - len(predictions))
            future_prices = np.array(future_prices[-days_ahead:])
        else:
            # No predictions available
            default_price = 80.0 if commodity == 'brent' else 65.0
            future_prices = np.array([default_price] * days_ahead)
    
    # Ensure no NaN values
    if np.isnan(future_prices).any():
        future_prices = replace_nan_values(future_prices, commodity=commodity)
    
    print(f"Future prices: {future_prices}")
    return future_prices


def generate_prediction_plot(last_date, future_dates, future_prices, 
                            historical_data, predictions, true_values, 
                            days_ahead, folder_path, commodity='brent'):
    """Generate visualization of historical data and predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data if available
    if 'dates' in historical_data and 'prices' in historical_data and len(historical_data['prices']) > 0:
        # Convert dates to matplotlib format if needed
        hist_dates = historical_data['dates']
        hist_prices = historical_data['prices']
        
        print(f"Plotting {len(hist_prices)} historical prices for {commodity}")
        print(f"Historical price range: {np.min(hist_prices):.2f} to {np.max(hist_prices):.2f}")
        
        plt.plot(hist_dates, hist_prices, 
                'b-', label='Historical Data', linewidth=2)
    
    # Plot forecasted data
    print(f"Plotting {len(future_prices)} future prices: {future_prices}")
    plt.plot(future_dates, future_prices, 'r--o', label='Forecasted Prices', 
             linewidth=2, markersize=8)
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, 
                label='Forecast Start')
    
    # Add labels and styling
    commodity_name = commodity.upper()
    title_text = f'{commodity_name} Price Forecast (Next {days_ahead} Days)\n'
    title_text += f'Based on Historical Data until {last_date.strftime("%Y-%m-%d")}'
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{commodity_name} Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.xticks(rotation=45)
    
    # Adjust y-axis limits for better visualization
    if 'prices' in historical_data and len(historical_data['prices']) > 0:
        all_prices = np.concatenate([historical_data['prices'], future_prices])
        y_min = np.min(all_prices) * 0.95
        y_max = np.max(all_prices) * 1.05
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=100)
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.read()).decode('utf-8')
    plt.close()
    
    return {
        "plot_base64": plot_base64,
        "historical_data": historical_data['data']
    }


def calculate_statistics(historical_data, future_prices, commodity='brent'):
    """Calculate statistics about the prediction"""
    # Get last known price
    if 'df' in historical_data and 'price_column' in historical_data:
        price_column = historical_data['price_column']
        last_known_price = historical_data['df'][price_column].values[-1]
        if np.isnan(last_known_price):
            valid_prices = historical_data['df'][price_column].dropna().values
            last_known_price = valid_prices[-1] if len(valid_prices) > 0 else (80.0 if commodity == 'brent' else 65.0)
    else:
        last_known_price = 80.0 if commodity == 'brent' else 65.0  # Default based on commodity
    
    # Calculate prediction statistics
    avg_forecasted_price = np.mean(future_prices)
    min_forecasted_price = np.min(future_prices)
    max_forecasted_price = np.max(future_prices)
    
    # Calculate percent change
    if last_known_price != 0:
        price_change = ((avg_forecasted_price - last_known_price) / last_known_price) * 100
    else:
        price_change = 0
    
    return {
        "last_known_price": float(last_known_price),
        "avg_forecasted_price": float(avg_forecasted_price),
        "min_forecasted_price": float(min_forecasted_price),
        "max_forecasted_price": float(max_forecasted_price),
        "price_change_percent": float(price_change)
    }


def prepare_past_predictions(last_date, predictions, true_values, days_ahead, commodity='brent'):
    """Prepare past predictions data for the frontend"""
    # Calculate how many past days we can show
    total_predictions = len(predictions)
    if total_predictions <= days_ahead:
        return []  # Not enough data for past predictions
    
    # Get up to 30 past predictions
    past_days = min(30, total_predictions - days_ahead)
    
    # Calculate start and end indices
    end_index = total_predictions - days_ahead
    start_index = max(0, end_index - past_days)
    
    # Extract past predictions and actuals
    past_predictions = predictions[start_index:end_index]
    past_actuals = true_values[start_index:end_index]
    
    # Generate past dates
    past_dates = [last_date - timedelta(days=past_days-i) for i in range(len(past_predictions))]
    
    # Ensure no NaNs
    past_predictions = replace_nan_values(past_predictions, commodity=commodity)
    past_actuals = replace_nan_values(past_actuals, commodity=commodity)
    
    # Create data for frontend
    past_predictions_data = []
    for i in range(len(past_dates)):
        date_str = past_dates[i].strftime('%Y-%m-%d')
        past_predictions_data.append({
            'date': date_str,
            'predicted': float(past_predictions[i]),
            'actual': float(past_actuals[i])
        })
    
    return past_predictions_data