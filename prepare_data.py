import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_model_data(config, df):
    # Extract configuration parameters
    sequence_length = config['sequence_length']
    train_periods = config['train_dates']
    target_station = config['target_station']
    source_stations = config['source_stations']
    target_columns = [f"{target_station}_stage", f"{target_station}_discharge"]

    # Convert all data columns to float to ensure compatibility with PyTorch
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Normalize each station's discharge and stage based on the first training period
    first_period = train_periods[0]
    train_data = df[(df['date'] >= first_period['start']) & (df['date'] <= first_period['end'])]
    scalers = {}
    for col in df.columns[1:]:  # Skip 'date' column
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])  # Scale entire dataset
        scalers[col] = scaler  # Save scaler for each column

    # Prepare source columns for X_data (input) and target columns for y_data (output)
    source_columns = [f"{station}_stage" for station in source_stations] + \
                     [f"{station}_discharge" for station in source_stations]

    # Prepare sequences for the full dataset (for predictions)
    X_data, y_data = [], []
    for i in range(len(df) - sequence_length):
        X_data.append(df.iloc[i:i+sequence_length][source_columns].values)  # Use only source columns
        y_data.append(df.iloc[i+sequence_length][target_columns].values)    # Use target columns

    # Convert to numpy arrays and then to PyTorch tensors
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    X_tensor = torch.tensor(X_data)
    y_tensor = torch.tensor(y_data)

    # DataLoader for the full dataset for predictions
    full_data = TensorDataset(X_tensor, y_tensor)
    full_loader = DataLoader(full_data, batch_size=len(X_tensor), shuffle=False)

    # Prepare sequences for each specified training period
    train_X_data, train_y_data = [], []
    for period in train_periods:
        start_date, end_date = period['start'], period['end']
        train_indices = (df['date'] >= start_date) & (df['date'] <= end_date)
        train_df = df[train_indices].reset_index(drop=True)
        
        for i in range(len(train_df) - sequence_length):
            train_X_data.append(train_df.iloc[i:i+sequence_length][source_columns].values)
            train_y_data.append(train_df.iloc[i+sequence_length][target_columns].values)

    # Convert to PyTorch tensors for training
    train_X_tensor = torch.tensor(np.array(train_X_data, dtype=np.float32))
    train_y_tensor = torch.tensor(np.array(train_y_data, dtype=np.float32))
    train_data = TensorDataset(train_X_tensor, train_y_tensor)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)

    print("DataLoader ready for training!")

    # Return the necessary tensors and scalers
    train_tensors = (train_X_tensor, train_y_tensor)
    return train_tensors, train_loader, full_loader, scalers

