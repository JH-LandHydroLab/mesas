# %%
timeseries_duration = 1.0
timeseries_length = 100
dt = timeseries_duration / timeseries_length

pulse_start = 0.05
pulse_end = 0.15

C_tracer_input = 0.5
Q_steady = 1.0
Storage_vol = 0.1

# %%
import numpy as np
import pandas as pd

data_df = pd.DataFrame(index=np.arange(timeseries_length) * dt)
data_df["Q out [vol/time]"] = Q_steady
data_df["J in [vol/time]"] = Q_steady
data_df["Storage_vol"] = Storage_vol
data_df["C [conc]"] = 0.0
data_df.loc[pulse_start:pulse_end, "C [conc]"] = C_tracer_input
data_df.to_csv("data.csv")

# %%
from mesas.sas.model import Model

model = Model(data_df="data.csv", config="config.json")
model.run()
model.data_df.to_csv("data_with_results.csv")
# %%
