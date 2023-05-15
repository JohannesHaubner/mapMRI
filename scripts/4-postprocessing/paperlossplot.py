import os, pathlib, json
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from parse import parse


labels = iter([r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"])

dpi = 400
figsize= None

foldername = "normalized-outputs"


fig1, ax1 = plt.subplots(dpi=dpi, figsize=figsize)

runnames = ["./outputs/my_registration_1/", "./outputs/my_registration_2/", "./outputs/my_registration_3/", "./outputs/my_registration_4/"]


for runname in runnames:

    runname = pathlib.Path(runname)

    lossfile = runname / "loss.txt"
    l2lossfile = runname / "l2loss.txt"
    hyperparameterfile = runname / "hyperparameters.json"
    hyperparameters = json.load(open(hyperparameterfile))
    domain_size = np.product(hyperparameters["target.shape"])

    loss = np.genfromtxt(lossfile, delimiter=",")

    loss = loss[~ np.isnan(loss)]
    try:
        iters = list(range(iters[-1], len(loss) + iters[-1]))
    except NameError:
        iters = list(range(len(loss)))

    startloss = loss[0] / domain_size
       
    linestlyle="-"

    label = next(labels)

    loss[:] /= domain_size

    p = ax1.plot(iters, loss, linestyle=linestlyle, label=label)        

ax1.set_ylabel("Mismatch deformed-target  \n " + r" $\frac{1}{|\Omega|}\int(\phi_i-\phi_e)^2\, dx$")

ax1.set_xticks([100, 200, 300, 400, 500])

for ax in [ax1]:
    plt.sca(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("LBFGS iteration")
    plt.tight_layout()

plt.tight_layout()
plt.show()
