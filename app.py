import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import io
import base64
import joblib # To load your trained model
import numpy as np # Import numpy
import seaborn as sns # Import seaborn for heatmap
from statsmodels.tsa.seasonal import seasonal_decompose # Import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Import ACF/PACF functions

# Suppress Matplotlib GUI backend warning in a non-interactive environment
plt.switch_backend('Agg')

app = Flask(__name__)

# --- Model Comparison Data ---
# Your provided model comparison data
model_performance_data = {
    "Model": [
        "Ensemble Stacking", "Random Forest", "XGBoost", "Holt Linear", "LSTM",
        "SARIMA", "GRU", "Holt-Winters", "CNN", "Prophet", "SARIMAX", "ARIMA",
        "DTW Similarity"
    ],
    "RMSE": [
        443.106583, 463.252270, 649.103945, 814.527360, 943.106956,
        946.783672, 947.111867, 952.353503, 955.070924, 1459.665722,
        1521.610092, 1568.099986, 3739.662615
    ],
    "MAE": [
        357.258268, 376.533062, 486.896758, 673.982214, 751.753920,
        791.350282, 795.427752, 714.425367, 738.253953, 1308.560925,
        1183.579509, 1384.356190, 2817.624483
    ],
    "MAPE": [
        4.480386, 4.652351, 5.829393, 8.487385, 9.715019,
        9.901513, 10.325892, 9.278759, 9.481984, 16.122657,
        15.118548, 18.366296, 36.648705
    ]
}
df_performance = pd.DataFrame(model_performance_data)

# --- Load the Best Model ---
meta_learner_model = None
try:
    meta_learner_model = joblib.load('ensemble_stacking_model.pkl')
    print("Ensemble Stacking model loaded successfully!")
except FileNotFoundError:
    print("Error: 'ensemble_stacking_model.pkl' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- Helper function to generate plots as base64 images ---
def get_plot_base64(df, x_col, y_col, title, ylabel, ascending=True):
    """
    Generates a matplotlib bar plot from a DataFrame and returns it as a base64 encoded string.
    Uses a Seaborn color palette for bars based on performance.
    """
    df_sorted = df.sort_values(by=y_col, ascending=ascending)
    colors = sns.color_palette("viridis_r", n_colors=len(df_sorted))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_sorted[x_col], df_sorted[y_col], color=colors)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_base64

def get_actual_time_series_plot_base64():
    """
    Loads data from 'department_17_sales.csv' and generates a time series plot.
    Returns a tuple: (plot_base64_string, error_message_string)
    """
    try:
        df_sales = pd.read_csv('department_17_sales.csv')
        date_column_name = 'Date'
        value_column_name = 'Weekly_Sales'

        if date_column_name not in df_sales.columns:
            raise KeyError(f"Date column '{date_column_name}' not found.")
        if value_column_name not in df_sales.columns:
            raise KeyError(f"Sales column '{value_column_name}' not found.")

        df_sales[date_column_name] = pd.to_datetime(df_sales[date_column_name])
        df_sales = df_sales.sort_values(by=date_column_name)

        line_color = sns.color_palette("deep")[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_sales[date_column_name], df_sales[value_column_name], color=line_color, linewidth=2)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weekly Sales", fontsize=12)
        ax.set_title("Department 17 Weekly Sales Time Series", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return plot_base64, None
    except FileNotFoundError:
        error_msg = "Error: 'department_17_sales.csv' not found. Please ensure it's in the same directory as app.py."
        print(error_msg)
        return None, error_msg
    except KeyError as e:
        error_msg = f"Error: Missing expected column in 'department_17_sales.csv': {e}. Please check 'Date' and 'Weekly_Sales' column names."
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while processing 'department_17_sales.csv': {e}"
        print(error_msg)
        return None, error_msg

def get_data_exploration_plots_base64():
    """
    Generates a 2x2 grid of data exploration plots and returns them as base64 encoded strings.
    Uses Seaborn color palettes for enhanced aesthetics.
    """
    plots = {}
    error_msg = None
    try:
        df = pd.read_csv('department_17_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        required_cols = ['Weekly_Sales', 'Temperature']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in 'department_17_sales.csv' for data exploration plots.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        palette_colors = sns.color_palette("viridis", 4)

        # Plot 1: Time series of Weekly Sales
        axes[0, 0].plot(df.index, df['Weekly_Sales'], linewidth=1.5, color=palette_colors[0])
        axes[0, 0].set_title('Department 17 Weekly Sales Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Weekly Sales ($)')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Sales distribution
        axes[0, 1].hist(df['Weekly_Sales'], bins=30, alpha=0.7, color=palette_colors[1], edgecolor='black')
        axes[0, 1].set_title('Distribution of Weekly Sales', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Weekly Sales ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0], fmt=".2f", linewidths=.5)
        axes[1, 0].set_title('Correlation Matrix', fontsize=14, fontweight='bold')

        # Plot 4: External factors vs Sales (Temperature)
        axes[1, 1].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.6, color=palette_colors[2])
        axes[1, 1].set_title('Temperature vs Weekly Sales', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Temperature (°F)')
        axes[1, 1].set_ylabel('Weekly Sales ($)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        plots['main_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

    except FileNotFoundError:
        error_msg = "Error: 'department_17_sales.csv' not found for data exploration plots. Please ensure it's in the same directory as app.py."
        print(error_msg)
    except KeyError as e:
        error_msg = f"Error: Missing expected column for data exploration plots: {e}. Ensure 'Weekly_Sales' and 'Temperature' are present."
        print(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred while generating data exploration plots: {e}"
        print(error_msg)

    return plots, error_msg

def get_decomposition_plot_base64():
    """
    Performs seasonal decomposition on 'Weekly_Sales' and generates a 4-plot grid.
    Uses Seaborn color palette for each component.
    Returns (plot_base64_string, error_message_string).
    """
    try:
        df = pd.read_csv('department_17_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        if 'Weekly_Sales' not in df.columns:
            raise KeyError("'Weekly_Sales' column not found for decomposition.")

        if len(df) < 2 * 52:
            raise ValueError(f"Not enough data for seasonal decomposition (need at least {2*52} periods for period=52). Data has {len(df)} periods.")

        decomposition = seasonal_decompose(df['Weekly_Sales'].dropna(),
                                         model='additive',
                                         period=52)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition_colors = sns.color_palette("tab10", 4)

        decomposition.observed.plot(ax=axes[0], title='Original Time Series', color=decomposition_colors[0])
        decomposition.trend.plot(ax=axes[1], title='Trend', color=decomposition_colors[1])
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color=decomposition_colors[2])
        decomposition.resid.plot(ax=axes[3], title='Residual', color=decomposition_colors[3])

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return plot_base64, None
    except FileNotFoundError:
        error_msg = "Error: 'department_17_sales.csv' not found for decomposition."
        print(error_msg)
        return None, error_msg
    except KeyError as e:
        error_msg = f"Error: Missing expected column for decomposition: {e}."
        print(error_msg)
        return None, error_msg
    except ValueError as e:
        error_msg = f"Error during decomposition: {e}. Check data length or period."
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while generating decomposition plot: {e}"
        print(error_msg)
        return None, error_msg

def get_acf_pacf_plots_base64():
    """
    Generates ACF and PACF plots for original and differenced 'Weekly_Sales' series.
    Returns (plot_base64_string, error_message_string).
    """
    try:
        df = pd.read_csv('department_17_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        if 'Weekly_Sales' not in df.columns:
            raise KeyError("'Weekly_Sales' column not found for ACF/PACF plots.")

        original_series = df['Weekly_Sales'].dropna()
        # Calculate first-order differenced series
        differenced_series = original_series.diff().dropna()

        # Define required minimum lengths for lags=40
        min_len_original = 40 + 1 # For lags=40, need at least 41 points
        min_len_differenced = 40 # For lags=40, need at least 40 points in differenced series

        print(f"ACF/PACF Plotting - Original series length: {len(original_series)}")
        print(f"ACF/PACF Plotting - Differenced series length: {len(differenced_series)}")

        if len(original_series) < min_len_original:
            raise ValueError(f"Not enough data for original series ACF/PACF (need at least {min_len_original} points for lags=40). Has {len(original_series)}.")
        if len(differenced_series) < min_len_differenced:
            raise ValueError(f"Not enough data for differenced series ACF/PACF (need at least {min_len_differenced} points for lags=40). Has {len(differenced_series)}.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Original series
        plot_acf(original_series, ax=axes[0, 0], lags=40, title='ACF - Original Series')
        plot_pacf(original_series, ax=axes[0, 1], lags=40, title='PACF - Original Series')

        # Differenced series
        plot_acf(differenced_series, ax=axes[1, 0], lags=40, title='ACF - Differenced Series')
        plot_pacf(differenced_series, ax=axes[1, 1], lags=40, title='PACF - Differenced Series')

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("ACF/PACF plots generated successfully.")
        return plot_base64, None
    except FileNotFoundError:
        error_msg = "Error: 'department_17_sales.csv' not found for ACF/PACF plots."
        print(error_msg)
        return None, error_msg
    except KeyError as e:
        error_msg = f"Error: Missing expected column for ACF/PACF plots: {e}."
        print(error_msg)
        return None, error_msg
    except ValueError as e:
        error_msg = f"Error during ACF/PACF plot generation: {e}. Check data length or lags."
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while generating ACF/PACF plots: {e}"
        print(error_msg)
        return None, error_msg


def generate_simple_forecast_plot(num_periods):
    """
    Generates a plot of historical sales data and a simple extrapolated forecast.
    Uses Seaborn colors for historical and forecast lines.
    Returns (plot_base64_string, error_message_string).
    """
    try:
        df_sales = pd.read_csv('department_17_sales.csv')
        date_column_name = 'Date'
        value_column_name = 'Weekly_Sales'

        if date_column_name not in df_sales.columns:
            raise KeyError(f"Date column '{date_column_name}' not found.")
        if value_column_name not in df_sales.columns:
            raise KeyError(f"Sales column '{value_column_name}' not found.")

        df_sales[date_column_name] = pd.to_datetime(df_sales[date_column_name])
        df_sales = df_sales.sort_values(by=date_column_name)

        last_date = df_sales[date_column_name].iloc[-1]
        last_sales = df_sales[value_column_name].iloc[-1]

        if len(df_sales) >= 4:
            recent_sales = df_sales[value_column_name].tail(4)
            trend = (recent_sales.iloc[-1] - recent_sales.iloc[0]) / (len(recent_sales) - 1)
        else:
            trend = 0

        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=num_periods, freq='W')
        forecast_values = [last_sales + (i + 1) * trend for i, _ in enumerate(forecast_dates)]

        fig, ax = plt.subplots(figsize=(14, 7))
        
        forecast_colors = sns.color_palette("Paired", 2)
        
        ax.plot(df_sales[date_column_name], df_sales[value_column_name], label='Historical Weekly Sales', color=forecast_colors[1], linewidth=2)
        ax.plot(forecast_dates, forecast_values, label=f'Simple Forecast ({num_periods} periods)', color=forecast_colors[0], linestyle='--', marker='o', markersize=4)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weekly Sales ($)", fontsize=12)
        ax.set_title(f"Department 17 Weekly Sales: Historical and Simple Forecast for {num_periods} Weeks", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return plot_base64, None
    except FileNotFoundError:
        error_msg = "Error: 'department_17_sales.csv' not found. Cannot generate forecast plot."
        print(error_msg)
        return None, error_msg
    except KeyError as e:
        error_msg = f"Error: Missing expected column for forecast plot: {e}. Ensure 'Date' and 'Weekly_Sales' are present."
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while generating forecast plot: {e}"
        print(error_msg)
        return None, error_msg

# --- Flask Route for the Main Dashboard ---
@app.route('/')
def dashboard():
    rmse_plot = get_plot_base64(df_performance, "Model", "RMSE", "Model RMSE Comparison", "RMSE", ascending=True)
    mae_plot = get_plot_base64(df_performance, "Model", "MAE", "Model MAE Comparison", "MAE", ascending=True)
    mape_plot = get_plot_base64(df_performance, "Model", "MAPE", "Model MAPE Comparison", "MAPE", ascending=True)
    
    actual_ts_plot, ts_plot_error = get_actual_time_series_plot_base64()

    return render_template(
        'index.html',
        performance_data=df_performance.to_dict(orient='records'),
        rmse_plot=rmse_plot,
        mae_plot=mae_plot,
        mape_plot=mape_plot,
        actual_ts_plot=actual_ts_plot,
        ts_plot_error=ts_plot_error,
        best_model_loaded=meta_learner_model is not None
    )

# --- Flask Route for Data Exploration Page ---
@app.route('/data_exploration')
def data_exploration():
    plots, error = get_data_exploration_plots_base64()
    decomposition_plot, decomposition_error = get_decomposition_plot_base64()
    acf_pacf_plot, acf_pacf_error = get_acf_pacf_plots_base64()
    return render_template('data_exploration.html',
                           plots=plots,
                           error=error,
                           decomposition_plot=decomposition_plot,
                           decomposition_error=decomposition_error,
                           acf_pacf_plot=acf_pacf_plot,
                           acf_pacf_error=acf_pacf_error)

# --- New Flask Route for Prediction Page ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    forecast_plot = None
    forecast_error = None
    num_periods_input = 2 # Default periods to forecast

    if request.method == 'POST':
        try:
            num_periods_input = int(request.form.get('num_periods', 4))
            if num_periods_input <= 0:
                raise ValueError("Number of periods must be positive.")
            forecast_plot, forecast_error = generate_simple_forecast_plot(num_periods_input)
        except ValueError as e:
            forecast_error = f"Invalid input for number of periods: {e}"
        except Exception as e:
            forecast_error = f"An unexpected error occurred during forecasting: {e}"
    else: # GET request, show initial form
        forecast_plot, forecast_error = generate_simple_forecast_plot(num_periods_input)

    return render_template(
        'predict.html',
        forecast_plot=forecast_plot,
        forecast_error=forecast_error,
        num_periods_input=num_periods_input
    )


if __name__ == '__main__':
    app.run(debug=True)
