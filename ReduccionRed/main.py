import numpy as np
import pandas as pd
import networkx as nx
from bokeh.plotting import figure, show
# from bokeh.io import output_notebook  # en caso de que se use colab
from bokeh.layouts import row, column
from bokeh.models import BoxAnnotation
from pyproj import Transformer
from tqdm import tqdm


proyeccion_mercator = Transformer.from_crs("epsg:4326", "epsg:3857",always_xy=True)

# Constantes
SUBESTACION = 3
PGEN = "PGEN"
QGEN = "QGEN"
PLOAD = "PLOAD"
QLOAD = "QLOAD"
XPU = "X"
RPU = "R"
VPU = "VPU"
GISX = "POSX"
GISY = "POSY"
TIPO = "TYPE"
NOMBRE = "NAME"
FROM = "FROM"
TO = "TO"

# Funciones

def print_warnings(G):
  num_n = G.number_of_nodes()
  num_l = G.number_of_edges()
  print("|- Grafo con ",num_n," nodos y ", num_l, "lineas")
  if num_n < num_l + 1 :
    print("|- La red no es radial")
    c = list(nx.simple_cycles(G.to_undirected()))
    if len(c) == 1: print("|- Bucle : ", c)
    else: print("|- Número de bucles: ", len(c))
  if num_n > num_l + 1 :
    print("|- El grafo no esta conectado")
  return 0

def load_feeder(feeder):
  print("Cargando alimentador de archivo CSV")
  nodos = pd.read_csv(feeder+"/nodes.csv")
  lineas = pd.read_csv(feeder+"/lines.csv")
  print("|- Generando el grafo orientado")
  G = nx.DiGraph()
  for index,nod in nodos.iterrows():
    s_gen = nod[PGEN] + nod[QGEN]*1j
    s_dem = nod[PLOAD] + nod[QLOAD]*1j
    x_m,y_m = proyeccion_mercator.transform(nod[GISX],nod[GISY])
    G.add_node(nod[NOMBRE],s_gen=s_gen, s_dem=s_dem, vpu = nod[VPU], tipo = nod[TIPO], gis_x = x_m, gis_y=y_m)
  for index,lin in lineas.iterrows():
    n1 = lin[FROM]
    n2 = lin[TO]
    zL = lin[RPU] + lin[XPU]*1j
    if zL!=zL : zL = 0
    G.add_edge(n1,n2,zL = zL)
  print_warnings(G)
  return G

def reduce_edges(G):
  G_new = G.copy()
  bk = [(n1,n2) for n1,n2,d in G_new.edges(data=True) if np.abs(d["zL"])==0]
  nl = 0
  while len(bk)>0:
    n1,n2 = bk[0]
    nl += 1
    G_new.remove_edge(n1,n2)
    s_gen = G_new.nodes[n1]["s_gen"] +  G_new.nodes[n2]["s_gen"]
    s_dem = G_new.nodes[n1]["s_dem"] +  G_new.nodes[n2]["s_dem"]
    G_new = nx.relabel_nodes(G_new, {n1:n2})
    G_new.nodes[n2]["s_gen"] = s_gen
    G_new.nodes[n2]["s_dem"] = s_dem
    bk = [(n1,n2) for n1,n2,d in G_new.edges(data=True) if np.abs(d["zL"])==0]
  num_n = G_new.number_of_nodes()
  num_l = G_new.number_of_edges()
  print("Eliminando ",nl," enlaces con impedancia cero")
  print_warnings(G_new)
  return G_new

def identify_loops(G):
  G_new = G.copy()
  L = list(nx.simple_cycles(G.to_undirected()))
  nl = 0
  for c in L:
    if len(c) == 1:
      G_new.remove_edge(c[0],c[0])
      nl += 1
  print("Eliminando ",nl," bucles unitarios")
  print_warnings(G_new)
  return G_new

def func_select_nodos(edge1,edge2):
  # selecciona el nodo a eliminar y los dos nodos
  n = edge1[0]             # caso1: edge1(n,n1) edge2(n,n2)
  n1 = edge1[1]
  n2 = edge2[1]
  if edge1[0] == edge2[1]: # caso2: edge1(n,n1) edge2(n2,n)
    n = edge1[0]
    n1 = edge1[1]
    n2 = edge2[0]
  if edge1[1] == edge2[0]: # caso3: edge1(n1,n) edge2(n,n2)
    n = edge1[1]
    n1 = edge1[0]
    n2 = edge2[1]
  if edge1[1] == edge2[1]: # caso4: edge1(n1,n) edge2(n2,n)
    n = edge1[1]
    n1 = edge1[0]
    n2 = edge2[0]
  return n, n1, n2

def reduce_intermediate_nodes(G):
  print("Eliminando nodos de paso")
  Gu = G.to_undirected()
  nr = [n for n in Gu.nodes() if len(Gu.edges(n))==2]
  while len(nr)>0:
    nl = nr[0]
    ed = list(Gu.edges(nl, data=True))
    lin1 = ed[0]
    lin2 = ed[1]
    zL = lin1[2]["zL"] + lin2[2]["zL"]
    n, n1, n2 = func_select_nodos([lin1[0],lin1[1]],[lin2[0],lin2[1]])
    Gu.remove_edge(lin1[0],lin1[1])
    Gu.remove_edge(lin2[0],lin2[1])
    s_gen = Gu.nodes[n]["s_gen"]
    s_dem = Gu.nodes[n]["s_dem"]
    Gu.nodes[n2]["s_gen"] += s_gen
    Gu.nodes[n2]["s_dem"] += s_dem
    Gu.remove_node(n)
    Gu.add_edge(n1,n2,zL = zL)
    nr = [n for n in Gu.nodes() if len(Gu.edges(n))==2]
  G_new = nx.DiGraph()
  for n,d in Gu.nodes(data=True):
    G_new.add_node(n,s_gen=d["s_gen"],s_dem=d["s_dem"],tipo=d["tipo"],vpu=d["vpu"],gis_x=d["gis_x"],gis_y=d["gis_y"])
  for n1,n2,d in Gu.edges(data=True):
    G_new.add_edge(n1,n2,zL = d["zL"])
  print_warnings(G_new)
  return G_new

def reduce_terminal_nodes(G):
  print("Eliminando nodos terminales (el resultado es una aproximación)")
  Gu = G.to_undirected()
  nr = [n for n,d in Gu.nodes(data=True) if (len(Gu.edges(n))==1)&(d["tipo"]!=SUBESTACION)]
  for n in nr:
    ed = list(Gu.edges(n, data=True))
    lin = ed[0]
    if lin[0] == n:
      nr = lin[1]
    else:
      nr = lin[0]
    zL = lin[2]["zL"]
    s_gen = Gu.nodes[n]["s_gen"]
    s_dem = Gu.nodes[n]["s_dem"]
    Iline = np.abs(s_gen-s_dem)/np.abs(Gu.nodes[n]["vpu"])
    s_per = zL*Iline**2
    Gu.remove_edge(lin[0],lin[1])
    Gu.nodes[nr]["s_gen"] += s_gen
    Gu.nodes[nr]["s_dem"] += s_dem+s_per
    Gu.remove_node(n)
  G_new = nx.DiGraph()
  for n,d in Gu.nodes(data=True):
    G_new.add_node(n,s_gen=d["s_gen"],s_dem=d["s_dem"],tipo=d["tipo"],vpu=d["vpu"],gis_x=d["gis_x"],gis_y=d["gis_y"])
  for n1,n2,d in Gu.edges(data=True):
    G_new.add_edge(n1,n2,zL = d["zL"])
  print_warnings(G_new)
  return G_new

def plot_feeder(G):
  print("Graficando alimentador")
  x = nx.get_node_attributes(G,"gis_x")
  y = nx.get_node_attributes(G,"gis_y")
  xl = [x[n] for n in x]
  yl = [y[n] for n in y]
  fig_gis = figure(x_range=(np.min(xl),np.max(xl)),
                   y_range=(np.min(yl),np.max(yl)),
                   x_axis_type = "mercator", y_axis_type="mercator")
  fig_gis.add_tile("CARTODBPOSITRON")
  fig_gis.scatter(x=xl,y=yl,size=5)
  for n1,n2 in G.edges():
    fig_gis.line(x=[x[n1],x[n2]],y=[y[n1],y[n2]])
  for n,d in G.nodes(data=True):
    if np.abs(d["s_dem"])>0 : fig_gis.scatter(x=x[n],y=y[n],size=5,color="black")
    if d["tipo"] == SUBESTACION : fig_gis.scatter(x=x[n],y=y[n],size=6,color="red")
  return fig_gis

def plot_vector(y):
  fig_v = figure(width=1200,height=200)
  fig_v.scatter(x=list(range(len(y))), y = y)
  return fig_v

def plot_hist(y):
  h_freq, h_marc = np.histogram(y)
  fig_h = figure(width=1200,height=200)
  fig_h.quad(bottom=0, top=h_freq, left = h_marc[1:], right = h_marc[:-1])
  return fig_h

def calculate_ybus(G):
  print("|- Calculando la Ybus")
  num_nodos = G.number_of_nodes()
  yp = np.diag([1/d["zL"] for n1,n2,d in G.edges(data=True)])
  A = nx.incidence_matrix(G,oriented=True)
  ybus = A @ yp @ A.T
  return ybus

def load_flow(G):
  print("Calculando flujo de carga")
  num_n = G.number_of_nodes()
  nodo_slack = [n for n,d in G.nodes(data=True) if d["tipo"] == SUBESTACION]
  nodos_todos = list(G.nodes())
  n_s = [nodos_todos.index(nodo_slack[0])] # indice del nodo slack
  n_r = np.setdiff1d(list(range(num_n)),n_s)
  v_slack = list([d["vpu"] for n,d in G.nodes(data=True) if d["tipo"]==SUBESTACION])
  ybus = calculate_ybus(G)
  v = np.ones(num_n)*(v_slack[0]+0j)
  sbus = np.array([d["s_gen"]-d["s_dem"] for n,d in G.nodes(data=True)])
  ynn = ybus[np.ix_(n_r,n_r)]
  iss = ybus[np.ix_(n_r,n_s)]@v[np.ix_(n_s)]
  iter = 0
  err = 1
  while (iter<30)&(err>1E-9):
    inn = np.conj(np.divide(sbus[np.ix_(n_r)],v[np.ix_(n_r)]))-iss
    v[np.ix_(n_r)] = np.linalg.solve(ynn,inn)
    iter += 1
    sc = np.multiply(np.conj(ybus@v),v)
    err = np.linalg.norm(sc[np.ix_(n_r)]-sbus[np.ix_(n_r)],np.inf)
  print("|- Despues de ",iter," iteraciones el error es de ",err )
  return v


# Cargar alimentador
feeder = "IEEE900Balanceado"
G = load_feeder(feeder)
fig1 = plot_feeder(G)
fig1.title = "Alimentador con "+str(G.number_of_nodes())+" nodos"

# Reducir tamaño del sistema
G = reduce_edges(G)
G = identify_loops(G)
G = reduce_intermediate_nodes(G)
fig2 = plot_feeder(G)
fig2.title = "Alimentador con "+str(G.number_of_nodes())+" nodos"


vn = load_flow(G)
fig3 = plot_vector(np.abs(vn))
fig3.title = "Voltajes nodales"
show(column(row(fig1,fig2),fig3))
