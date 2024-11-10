import numpy as np
import pandas as pd
import glob

def load_project_data(data_dir):

    # Define file paths for stage and discharge
    stage_files = glob.glob(f'{data_dir}/*_stage.txt')
    discharge_files = glob.glob(f'{data_dir}/*_discharge.txt')

    # Initialize a list to store each file's data before combining
    all_data = []

    # Load and structure stage data
    for file_path in stage_files:
        df = pd.read_csv(
            file_path,
            header=None,
            names=['date', 'value'],
            sep='\s+',
            parse_dates=['date'],
            date_format='%Y-%m-%d'
        )
        station_name = file_path.split('/')[-1].replace('_stage.txt', '')
        df['station'] = station_name
        df['variable_type'] = 'stage'
        all_data.append(df)

    # Load and structure discharge data
    for file_path in discharge_files:
        df = pd.read_csv(
            file_path,
            header=None,
            names=['date', 'value'],
            sep='\s+',
            parse_dates=['date'],
            date_format='%Y-%m-%d'
        )
        station_name = file_path.split('/')[-1].replace('_discharge.txt', '')
        df['station'] = station_name
        df['variable_type'] = 'discharge'
        all_data.append(df)

    # Combine all data into a single DataFrame
    df = pd.concat(all_data)

    # Pivot the data to have a wide format
    df = df.pivot_table(
        index='date', 
        columns=['station', 'variable_type'], 
        values='value'
    )

    # Flatten the MultiIndex columns for easier access
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)

    # Replace NaN values with zero for H3_Huiku-61505100_discharge
    df.loc[:, 'H3_Huiku-61505100_discharge'] = df['H3_Huiku-61505100_discharge'].fillna(0)

    # Replace NaN values in H3_Huiku-61505100_stage with the minimum value of the column
    df.loc[:, 'H3_Huiku-61505100_stage'] = df['H3_Huiku-61505100_stage'].fillna(np.nanmin(df['H3_Huiku-61505100_stage']))

    return df
