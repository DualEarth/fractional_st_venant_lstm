import pandas as pd
import matplotlib.pyplot as plt

def plot_project_data(df):

    # Plot all data on a single chart with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 4))
    plt.xticks(rotation=45)

    # Plot discharge data on the primary y-axis
    for col in df.columns:
        if 'discharge' in col:
            ax1.plot(df['date'], df[col], label=f'Discharge - {col.split("_")[0]}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Discharge (cfs)')
    ax1.tick_params(axis='y')

    # Create a secondary y-axis for stage data
    ax2 = ax1.twinx()
    for col in df.columns:
        if 'stage' in col:
            ax2.plot(df['date'], df[col], linestyle='--', label=f'Stage - {col.split("_")[0]}')
    ax2.set_ylabel('Stage (ft)')
    ax2.set_ylim([25,40])
    ax2.tick_params(axis='y')

    # Combine legends from both y-axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Show grid and plot
    plt.title('Discharge and Stage Time Series for All Stations')
    plt.grid(True)
    plt.show()
    plt.close()