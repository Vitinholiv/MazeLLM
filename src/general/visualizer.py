import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.colors import ListedColormap

matrix = np.zeros((20, 20))
matrix[:, 0] = 1

cmap = ListedColormap(['lightgray', 'limegreen'])

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)

img = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
ax.set_title("Active Column: 1")

ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
slider = Slider(ax_slider, 'Column', 1, 20, valinit=1, valstep=1)

def update(val):
    col_idx = int(slider.val) - 1
    
    new_matrix = np.zeros((20, 20))
    new_matrix[:, col_idx] = 1
    
    img.set_data(new_matrix)
    ax.set_title(f"Active Column: {int(slider.val)}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()