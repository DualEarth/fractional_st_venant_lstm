import torch
from torch.utils.data import DataLoader, TensorDataset
from models import LSTM_combo_Model, LSTMModel
import numpy as np

def train_and_run_model(config, data):
    
    if config['model_name'] == "LSTMModel":
        model = LSTMModel()
    if config['model_name'] == "LSTM_combo_Model":
        model = LSTM_combo_Model()

    train_tensors, train_loader, full_loader, scalers = data
    train_X_tensor, train_y_tensor = train_tensors
    target_station = config['target_station']

    # Extract mean/std values for un-normalizing target predictions
    target_stage_mean = scalers[f"{target_station}_stage"].mean_
    target_stage_std = scalers[f"{target_station}_stage"].scale_
    target_discharge_mean = scalers[f"{target_station}_discharge"].mean_
    target_discharge_std = scalers[f"{target_station}_discharge"].scale_

    num_epochs = config['num_epochs']
    status_interval = config['status_interval']  # Print training status every 10 epochs
    # Loop over multiple model instances for ensemble training
    num_ensembles = config['num_ensembles']
    ensemble_predictions = []

    criterion = torch.nn.MSELoss()

    for model_num in range(num_ensembles):
        # Initialize a new model for each ensemble member
#        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        
        # Train the model
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Print training status at specified intervals
            if (epoch + 1) % status_interval == 0:
                print(f"Model {model_num + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Make predictions after training and save in ensemble_predictions list
        with torch.no_grad():
            for X_full, _ in full_loader:
                y_preds = model(X_full).cpu().numpy()
            
            # Un-normalize predictions
            y_preds_unscaled = np.empty_like(y_preds)
            y_preds_unscaled[:, 0] = (y_preds[:, 0] * target_stage_std) + target_stage_mean
            y_preds_unscaled[:, 1] = (y_preds[:, 1] * target_discharge_std) + target_discharge_mean
            
            # Store final predictions for each ensemble model
            ensemble_predictions.append(y_preds_unscaled)

    return ensemble_predictions