# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mesas.sas.model import Model

# def test_evapoconcentration():
if True:
    timeseries_length = 10
    max_age = timeseries_length
    dt = 0.1
    Q_0 = 1.0  # <-- steady-state flow rate
    E_0 = 1.0  # <-- steady-state flow rate
    C_J = 1.0

    C_old = 1.0
    J = Q_0 + E_0
    S_0 = 5 * Q_0
    S_m = 0.0
    n_substeps = 1
    debug = False
    verbose = True
    jacobian = False
    num_scheme = 4

    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["t"] = data_df.index * dt
    data_df["J"] = J
    data_df["Q"] = Q_0
    data_df["E"] = E_0
    data_df["S_0"] = S_0
    data_df["S_m"] = S_m
    data_df["C"] = C_J

    solute_parameters = {
        "C": {
            "C_old": C_old,
            "alpha": {"Q": 1.0, "E": 0.0},
        }
    }
    sas_specs = {
        "Q": {
            "spec": {"ST": ["S_m", "S_0"]},
        },
        "E": {
            "spec": {"ST": ["S_m", "S_0"]},
        },
    }

    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_parameters,
        debug=debug,
        verbose=verbose,
        dt=dt,
        n_substeps=n_substeps,
        jacobian=jacobian,
        max_age=max_age,
        warning=True,
        num_scheme=num_scheme,
        record_state=True,
    )
    model.run()
    data_df = model.data_df

    C_J = data_df["C"]
    pQ = model.get_pQ(flux="Q")
    C_Q = np.zeros(timeseries_length)

    CT = model.get_CT("C")
    new_CT = np.ones_like(CT)

    # Update the first row of new_CT
    new_CT[0] = (CT[0] + 1.0) / 2.0

    # Update the remaining elements
    for i in range(1, CT.shape[0]):
        for j in range(i, CT.shape[1]):
            new_CT[i, j] = (CT[i - 1, j - 1] + CT[i, j]) / 2.0

    evapoconc_factor = new_CT[:, 1:]
    evapoconc_factor[np.isnan(evapoconc_factor)] = 1.0
    for t in range(timeseries_length):
        # the maximum age is t
        for T in range(t + 1):
            # the entry time is ti
            ti = t - T
            C_Q[t] += C_J[ti] * pQ[T, t] * evapoconc_factor[T, t] * dt

        C_Q[t] += C_old * (1 - pQ[: t + 1, t].sum() * dt)

    plt.figure()
    plt.plot(C_Q, label="C_Q from convolution")
    plt.plot(data_df["C --> Q"], label="C --> Q")
    plt.legend(frameon=False)
# %%
