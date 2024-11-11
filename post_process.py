import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def process_model_data(config, df, model_predictions):
    # Combine predictions from all ensemble members into plot_df and add target columns
    prediction_dates = df['date'].iloc[config['sequence_length']:].reset_index(drop=True)
    plot_df = pd.DataFrame({'date': prediction_dates})

    # Add ensemble predictions to plot_df
    for i, preds in enumerate(model_predictions):
        plot_df[f'pred_stage_e{i+1}'] = preds[:, 0]  # Stage predictions
        plot_df[f'pred_discharge_e{i+1}'] = preds[:, 1]  # Discharge predictions

    # Calculate ensemble mean
    plot_df['pred_stage'] = plot_df[[f'pred_stage_e{i+1}' for i in range(config['num_ensembles'])]].mean(axis=1)
    plot_df['pred_discharge'] = plot_df[[f'pred_discharge_e{i+1}' for i in range(config['num_ensembles'])]].mean(axis=1)

    # Set date as the index
    plot_df.set_index('date', inplace=True)

    # Calculate total training days across all periods in `train_dates`
    total_days = 0
    for period in config['train_dates']:
        start_date = datetime.strptime(period['start'], '%Y-%m-%d')
        end_date = datetime.strptime(period['end'], '%Y-%m-%d')
        total_days += (end_date - start_date).days + 1  # +1 to include both start and end dates

    # Construct the filename using the model name and total training days
    model_name = config['model_name']
    file_name = f"./output/{model_name}_training_{total_days}_days.csv"

    # Export the DataFrame
    export_df = plot_df.loc[:, ['pred_stage', 'pred_discharge']]
    export_df.to_csv(file_name)

    print(f"Data saved to {file_name}")
    return plot_df

def plot_results(config, df, plot_df, data):

    train_tensors, train_loader, full_loader, scalers = data
    train_X_tensor, train_y_tensor = train_tensors
    # Extract mean/std values for un-normalizing target predictions
    target_station = config['target_station']
    target_columns = [f"{target_station}_stage", f"{target_station}_discharge"]
    target_stage_mean = scalers[f"{target_station}_stage"].mean_
    target_stage_std = scalers[f"{target_station}_stage"].scale_
    target_discharge_mean = scalers[f"{target_station}_discharge"].mean_
    target_discharge_std = scalers[f"{target_station}_discharge"].scale_

    # Convert plot_start_date and plot_end_date to datetime objects
    plot_start_date = datetime.strptime(config['plot_start_date'], "%Y-%m-%d")
    plot_end_date = datetime.strptime(config['plot_end_date'], "%Y-%m-%d")

    # Filter original combined_df for the specified date range for source data
    df = df.set_index("date")
    source_df = df.loc[plot_start_date:plot_end_date]
    df = df.reset_index()

    # Define source columns
    source_stages = [f"{station}_stage" for station in config['source_stations']]
    source_discharges = [f"{station}_discharge" for station in config['source_stations']]

    # Unscale the source columns for plotting using the scalers from the training period
    for col in source_stages + source_discharges:
        source_df[col] = source_df[col] * scalers[col].scale_ + scalers[col].mean_
        

    # Extract the target columns from df for the entire range (for visualization)
    target_stage = df[target_columns[0]].iloc[config['sequence_length']:].values
    target_discharge = df[target_columns[1]].iloc[config['sequence_length']:].values
    plot_df['target_stage'] = target_stage
    plot_df['target_discharge'] = target_discharge

    # Unscale the target stage and discharge in plot_df using saved scalers
    plot_df['target_stage'] = (plot_df['target_stage'] * target_stage_std) + target_stage_mean
    plot_df['target_discharge'] = (plot_df['target_discharge'] * target_discharge_std) + target_discharge_mean

    # Plotting ensemble member predictions for stage and discharge
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Stage Plot ---

    # Plot target stage
    ax1.plot(plot_df.index, plot_df['target_stage'], label='H13 Stage groundtruth', color='k', linestyle='--', lw=3)

    # Plot each source station's stage data from source_df
    for source_stage in source_stages:
        ax1.plot(source_df.index, source_df[source_stage], linestyle='-.', lw=1.5, label=f'{source_stage.split("_")[0]} Stage')

    # Plot individual ensemble member predictions in grey
    for member in range(config['num_ensembles']):
        if member == 0:
            ax1.plot(plot_df.index, plot_df[f'pred_stage_e{member+1}'], color='grey', alpha=0.1, label="ensemble member")
        else:
            ax1.plot(plot_df.index, plot_df[f'pred_stage_e{member+1}'], color='grey', alpha=0.1)

    # Plot ensemble mean prediction in blue
    ax1.plot(plot_df.index, plot_df['pred_stage'], color='blue', lw=2, label='H13 Stage prediction')

    # Set x-axis limits
    ax1.set_xlim([plot_start_date, plot_end_date])

    # Add labels and legend for stage
    ax1.set_ylabel("Stage", color="navy")
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1))
    ax1.set_title("LSTM Model Stage Predictions with Ensemble Members and Mean")

    # --- Discharge Plot ---

    # Plot target discharge
    ax2.plot(plot_df.index, plot_df['target_discharge'], label='H13 Discharge groundtruth', color='k', linestyle='--', lw=3)

    # Plot each source station's discharge data from source_df
    for source_discharge in source_discharges:
        ax2.plot(source_df.index, source_df[source_discharge], linestyle='-.', lw=1.5, label=f'{source_discharge.split("_")[0]} Discharge')

    # Plot individual ensemble member discharge predictions in grey
    for member in range(config['num_ensembles']):
        if member == 0:
            ax2.plot(plot_df.index, plot_df[f'pred_discharge_e{member+1}'], color='grey', alpha=0.1, label="ensemble member")
        else:
            ax2.plot(plot_df.index, plot_df[f'pred_discharge_e{member+1}'], color='grey', alpha=0.1)

    # Plot ensemble mean prediction in blue for discharge
    ax2.plot(plot_df.index, plot_df['pred_discharge'], color='blue', lw=2, label='H13 Discharge prediction')

    # Add labels and legend for discharge
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Discharge", color="darkred")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 1))
    ax2.set_title("LSTM Model Discharge Predictions with Ensemble Members and Mean")

    plt.tight_layout()
    plt.show()


def plot_all_results(df):
    # Path to output files
    output_dir = "./output/"
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

    # Load plot start and end dates
    plot_start_date = datetime.strptime("2021-04-01", "%Y-%m-%d")
    plot_end_date = datetime.strptime("2021-10-01", "%Y-%m-%d")

    # Filter `df` for the target date range
    df = df.set_index('date')
    target_discharge = df.loc[plot_start_date:plot_end_date, 'H13_Anxiang-61505900_discharge']
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

    for i, file_path in enumerate(csv_files):
        # Load each model's prediction
        model_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        
        # Filter based on the plot range and align with target_discharge
        model_df = model_df.loc[plot_start_date:plot_end_date]
        aligned_target_discharge = target_discharge.reindex(model_df.index)

        # Ensure lengths match by dropping NaNs in both series
        aligned_target_discharge = aligned_target_discharge.dropna()
        aligned_pred_discharge = model_df['pred_discharge'].reindex(aligned_target_discharge.index).dropna()

        # Extract model name from the file name
        model_name = os.path.basename(file_path).replace(".csv", "")
        
        # Plot discharge predictions
        ax.plot(aligned_pred_discharge.index, aligned_pred_discharge, label=model_name, color=colors[i], lw=1.5)

    # Plot the target discharge data
    ax.plot(aligned_target_discharge.index, aligned_target_discharge, label="H13_Anxiang-61505900_discharge", color='black', linestyle="--", lw=2)

    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge")
    ax.set_title("Model Discharge Predictions")

    # Place legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    plt.tight_layout()
    plt.show()

def calculate_metrics(df):
    # Path to output files
    output_dir = "./output/"
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

    # Load plot start and end dates
    plot_start_date = datetime.strptime("2021-04-01", "%Y-%m-%d")
    plot_end_date = datetime.strptime("2021-10-01", "%Y-%m-%d")

    # Filter `df` for the target date range
    df = df.set_index('date')
    target_discharge = df.loc[plot_start_date:plot_end_date, 'H13_Anxiang-61505900_discharge']

    # Initialize metrics table
    metrics_data = []

    for file_path in csv_files:
        # Load each model's prediction
        model_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        
        # Filter based on the plot range and align with target_discharge
        model_df = model_df.loc[plot_start_date:plot_end_date]
        aligned_target_discharge = target_discharge.reindex(model_df.index)

        # Ensure lengths match by dropping NaNs in both series
        aligned_target_discharge = aligned_target_discharge.dropna()
        aligned_pred_discharge = model_df['pred_discharge'].reindex(aligned_target_discharge.index).dropna()

        # Calculate RMSE
        rmse = np.sqrt(np.mean((aligned_target_discharge - aligned_pred_discharge) ** 2))

        # Calculate R-squared (R2) manually
        ssr = np.sum((aligned_target_discharge - aligned_pred_discharge) ** 2)  # Sum of squares of residuals
        sst = np.sum((aligned_target_discharge - np.mean(aligned_target_discharge)) ** 2)  # Total sum of squares
        r2 = 1 - (ssr / sst) if sst != 0 else float('nan')  # Handling division by zero if variance is zero

        # Calculate NSE (using similar logic)
        nse = 1 - (ssr / sst)  # The NSE formula is equivalent to R2 when computed over means

        # Calculate KGE (for comparison)
        cc = np.corrcoef(aligned_target_discharge, aligned_pred_discharge)[0, 1]
        alpha = np.std(aligned_pred_discharge) / np.std(aligned_target_discharge)
        beta = np.mean(aligned_pred_discharge) / np.mean(aligned_target_discharge)
        kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # Extract model name from the file name
        model_name = os.path.basename(file_path).replace(".csv", "")
        
        # Append results to the table
        metrics_data.append({
            "Model": model_name,
            "RMSE": rmse,
            "R2": r2,
            "NSE": nse,
            "KGE": kge
        })

    # Convert metrics data to DataFrame and print
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)