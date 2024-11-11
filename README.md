# LSTM comparisons for the Fractional SVE paper

## How to run
This repository contains a model for comparing two LSTM architectures using hydrological time series data. To run the model, simply open the `notebook.ipynb` file and execute the cells. The notebook loads all required code from external .py files, including: 
  - `load_data.py`
  - `plot_data.py`
  - `prepare_data.py`
  - `train_and_run_model.py`
  - `post_process.py`
  - `models.py`  

The configuration files, `./config/*.yml`, are used to manage all customizable parameters. Key settings include model_name, which allows you to select either the "LSTMModel" or "LSTM_combo_Model" architecture, as well as train_dates, where you specify date ranges for training (you can uncomment or add multiple date ranges if needed). Additional parameters include target_station and source_stations for selecting specific station data, dropout_prob for controlling dropout regularization, sequence_length for setting input sequence length, and various training parameters such as num_epochs and learning_rate. You can also control ensemble model runs with num_ensembles and set the plot_start_date and plot_end_date for the visualization range.


## Model architectures
**LSTMModel:** is a straightforward LSTM architecture with an external dropout layer for regularization. It has a single LSTM layer that encodes the sequence input into a hidden state, which is then passed through a fully connected layer to produce the final output. This model is designed to capture temporal dependencies in the input data in a direct manner, mapping the encoded sequence to the target output.  

**LSTM_combo_Model:** extends the basic LSTM architecture by incorporating a mechanism to weight the final output based on the input data's last timestep. This model includes an LSTM layer that processes the input sequence and outputs a hidden state. After dropout regularization, the hidden state is passed through a fully connected layer that generates weights for each of the input features in the last timestep. These weights are then applied to the final timestep values of the input features (stages and discharges from multiple sources), producing a weighted sum that is used to generate the model’s output. This approach enables the model to use both the temporal information captured by the LSTM and the current state of each source variable, providing a combined representation that is well-suited for tasks where the relationships between sources and target variables may vary dynamically across time.