import matplotlib.pyplot as plt

from mesas.sas.model import Model

eg_list = ["lower_hafren", "hyporheic"]

# def test_examples(makeplots=False):
#    for eg in eg_list:
#        run_example(eg, makeplots)


def run_example(eg, makeplots=False):
    # Create the model
    model = Model(data_df=f"./examples/{eg}/data.csv", config=f"./examples/{eg}/config.json")

    # Run the model
    model.run()

    # Extract results
    data_df = model.data_df
    flux = model.fluxorder[0]

    # make plots
    fig = plt.figure()
    print(model.solorder)
    for isol, sol in enumerate(model.solorder):
        print(isol, sol)
        C_pred = data_df[f"{sol} --> {flux}"]
        C_obs = data_df[model.solute_parameters[sol]["observations"]]
        ax = plt.subplot2grid((model._numsol, 1), (isol, 0))
        ax.plot(C_pred, "k-", lw=0.2, label="SAS model prediction")
        ax.plot(C_obs, "r-", marker=".", lw=0.5, ms=3, alpha=1, label="Observations")
        ax.set_title(sol)
    plt.tight_layout()
    fig.savefig(eg + ".pdf")


if __name__ == "__main__":
    for eg in eg_list:
        run_example(eg, True)
