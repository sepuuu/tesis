import re
import numpy as np
from scipy.ndimage import gaussian_filter  
import matplotlib.pyplot as plt

# Cargar imagen de fondo  
field = plt.imread('cancha.png')

# Extraer puntos
points = []
with open('posiciones_jugadores.txt') as f:
  for line in f: 
    coords = re.findall(r'\d+', line)
    x, y = map(int, coords[:2])
    points.append((x, y))
    
# Crear heatmap
heatmap = np.zeros((1080, 1920)) 
for x, y in points:
  heatmap[y:y+40, x:x+40] += 1
  
heatmap = gaussian_filter(heatmap, sigma=5)
heatmap = np.uint8(255 * (heatmap / heatmap.max()))

# Graficar imagen de fondo  
plt.imshow(field)

# Graficar heatmap sobre imagen
plt.imshow(heatmap, cmap='hot', alpha=0.7) 

# Desactivar ejes
plt.axis('off')  
plt.savefig('heatmap_plot.png', bbox_inches='tight', pad_inches=0)
plt.show()
