mport numpy as np
import pandas as pd
from bokeh.plotting import figure, show, row
from sklearn.cluster import KMeans


# Cargar la base de datos
print("Datos de radiación solar:")
data = pd.read_csv("SolarData.csv")
print(data)

# agrupar por dias
data_x_fecha = data.groupby("Fecha")
data_dia  = data_x_fecha.get_group("2/5/2006")
n = len(data_dia)

fig1 = figure()
for name,group in data_x_fecha:
    fig1.line(x=np.linspace(0,24,n),y=group["Potencia"])
fig1.line(x=np.linspace(0,24,n), y=data_dia["Potencia"],color="yellow")

# Clusterizacion
d = []
x = []
for name, group in data_x_fecha:
  d += [name]
  x += [np.array(group["Potencia"])]
x = np.array(x)

nc = 4
kmeans = KMeans(n_clusters = nc).fit(x)
scenarios = kmeans.cluster_centers_

from bokeh.palettes import Category10
colores = Category10[nc]
fig2 = figure()
for k in range(nc):
   fig2.line(x = np.linspace(0,24,24*12),
            y = scenarios[k][:],
            color=colores[k])
show(row(fig1,fig2))
