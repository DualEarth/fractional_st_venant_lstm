import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

    export_df = plot_df.loc[:,['pred_stage', 'pred_discharge']]
    export_df.to_csv("lstm_x_extra.csv")

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

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