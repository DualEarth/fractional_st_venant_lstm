import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=2, num_layers=1, dropout_prob=0.4):
        super(LSTMModel, self).__init__()
        
        # LSTM layer with dropout applied externally
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Dropout layer after LSTM output
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        # Fully connected layer to map the hidden state to the output
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass through LSTM; hn is the hidden state of the last timestep
        _, (hn, _) = self.lstm(x)
        
        # Apply dropout to the LSTM output (hidden state)
        lstm_out = self.dropout(hn[-1])  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer to get the final output
        out = self.fc(lstm_out)  # Shape: (batch_size, output_size)
        
        return out

class LSTM_combo_Model(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=6, num_layers=1, dropout_prob=0.4):
        super(LSTM_combo_Model, self).__init__()
        
        # LSTM layer without dropout applied internally
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Dropout layer to be applied after the LSTM's output
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        # Fully connected layer to produce weights for each input feature in the last timestep
        self.fc = torch.nn.Linear(hidden_size, output_size)  # Produces 6 weights (3 sources * 2 variables)
    
    def forward(self, x):
        # Pass through LSTM
        _, (hn, _) = self.lstm(x)
        
        # Apply dropout to the LSTM's output (hidden state)
        lstm_out = self.dropout(hn[-1])  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer to produce weights
        fc_out = self.fc(lstm_out)  # Shape: (batch_size, 6)
        
        # Reshape to match the source structure (batch_size, 3 sources, 2 variables)
        fc_out = fc_out.view(-1, 3, 2)  # Shape: (batch_size, 3, 2)
        
        # Extract the last timestep of the input sequence for each input source
        x_last_timestep = x[:, -1, :]  # Shape: (batch_size, input_size=6)
        
        # Reshape to match source structure for weighting
        x_sources = x_last_timestep.view(-1, 3, 2)  # Shape: (batch_size, 3, 2)
        
        # Apply weights from the LSTM output to the sources in the last timestep
        weighted_sources = fc_out * x_sources  # Element-wise multiplication, shape: (batch_size, 3, 2)
        
        # Sum along the source dimension to produce the final output
        final_out = weighted_sources.sum(dim=1)  # Shape: (batch_size, 2)

        return final_out
