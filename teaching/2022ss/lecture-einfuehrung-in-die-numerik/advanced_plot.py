import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 100, num = 1000)
y1 = np.sin(x * 0.2) + 1.2
y2 = np.cos(x) - 1

fig, axs = plt.subplots(figsize=(10,10))
axs.plot(x, y1, label="Line plot of sin(x) + 1.2", c = "black")
cax = axs.scatter(x, y2, label="Scatter plot of cos(x) - 1", c = y1)
axs.set_xlim(0, 100)
axs.set_ylim(-2.5, 2.5)
axs.set_title("The plot title")
axs.set_xlabel(r"x / $\mathrm{\AA}$")
axs.set_ylabel(r"y / $\mathrm{\frac{kg}{m³}}$")
plt.legend()
cbar = plt.colorbar(cax)
cbar.ax.set_ylabel("z / unit")
plt.show()
