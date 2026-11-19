# PM1 Forecasting — Data-Loss 
# LSTM Baseline Overview:
This repo contains the data-loss LSTM baseline (v-lstm.py) for forecasting PM1 concentrations, built as part of a broader Physics-Informed LSTM (PI-LSTM) project mentored by Dr. Anubhab Roy (IIT Madras). It trains on 5,868 samples across 192 aerosol size-bin features (4,684 train sequences, 1,172 test sequences) using data loss only (MSE), before adding physics loss terms.   

# Results & XGBoost Metric Conversion 
Across 20 epochs, training MSE dropped from 1.1341 to 0.7532, while test MSE hit 0.3138 at best (Epoch 6) and ended at 0.3454 (Epoch 20).To compare with our earlier Optuna-tuned XGBoost weather-station model (which brought MAE down to 5.32–5.89), we convert MSE to RMSE via square root (sqrt(MSE)):Epoch 20 Test MSE (0.3454): RMSE is about 0.59 (and rough MAE equivalent is around 0.47 on the model's scale, or scaled by your target's standard deviation once denormalized).   Best Test MSE at Epoch 6 (0.3138): RMSE is about 0.56 (rough MAE equivalent around 0.45 on the model's scale).   
# LSTM vs. XGBoost 
Sequence vs. Tabular: The LSTM models time sequences directly across 192 size bins, whereas XGBoost worked on tabular/lagged weather-station rows.   
# Short vs. Long Horizon Behavior: 
XGBoost had higher error on short 1-minute steps than 1-hour steps due to high-frequency noise, while this LSTM captures sequential dynamics across size-bin distributions in a clean sequence setup.   
# Current Loss: 
Purely data-driven MSE here—no physical equations enforced yet.   

# Future DirectionWind Sensors:
Add wind speed and wind direction data to separate local aerosol changes from wind transport and dispersion.
# Aerosol General Dynamic Equation (GDE): 
Add a GDE physics loss term alongside data loss in the full PI-LSTM model so predictions respect coagulation, condensation/evaporation, sources, and removal.
