import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)
y1, y2, y3 = x, x**2, x**3

# Figure erstellen
plt.figure()

# Plots erstellen
plt.plot(x, y1, label='linear')
plt.plot(x, y2, label='quadratisch')
plt.plot(x, y3, label='kubisch')

# Labels fuer x- und y-Achse
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')

# Titel, Legende, Grid
plt.title('Polynome')
plt.legend()
plt.grid()

# Plot anzeigen
plt.show()
