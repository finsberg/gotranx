"""Plot the potential along the cable over time, as written by main.cu (v.csv)."""

import sys

import matplotlib.pyplot as plt
import numpy as np

dx = 0.1  # mm, as in cable.cuh

data = np.loadtxt(sys.argv[1] if len(sys.argv) > 1 else "v.csv", delimiter=",")
t, v = data[:, 0], data[:, 1:]
x = np.arange(v.shape[1]) * dx

fig, ax = plt.subplots(figsize=(8, 5))
image = ax.imshow(
    v.T,
    aspect="auto",
    origin="lower",
    extent=(t[0], t[-1], x[0], x[-1]),
    cmap="viridis",
)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Position along the cable (mm)")
fig.colorbar(image, ax=ax, label="Membrane potential (mV)")
fig.tight_layout()
fig.savefig("cable.png", dpi=150)
