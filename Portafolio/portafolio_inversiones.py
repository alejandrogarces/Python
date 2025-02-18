import pandas as pd
import cvxpy as cv
import numpy as np

inversiones = pd.read_csv("inversiones.csv") 
presupuesto = pd.read_csv("presupuesto.csv")
total_inversiones = sum(inversiones["Valor"])
total_presupuesto = sum(presupuesto["Valor"])
presupuesto.loc[len(presupuesto)] = ["fuera del plan",total_inversiones]
num_inversiones = len(inversiones)
num_periodos = len(presupuesto)

x = cv.Variable((num_inversiones,num_periodos),integer=True)
f_inv = [0]*num_periodos
f_ben = [0]*num_periodos
res = []
for t in range(num_periodos):
    for k in range(num_inversiones):
        f_inv[t] += inversiones["Valor"][k]*x[k,t]
        t_ret = num_periodos-t-1
        f_ben[t] += inversiones["Beneficio Anual"][k]*x[k,t]*t_ret
        res += [x[k,t]>=0]
        res += [x[k,t]<=1]

for k in range(num_inversiones):
    res += [sum(x[k,:]) == 1.0]
for t in range(num_periodos):
    res += [f_inv[t] <= presupuesto["Valor"][t]]
modelo = cv.Problem(cv.Minimize(sum(f_inv)-sum(f_ben)),res)
modelo.solve()
print(modelo.status)
for t in range(num_periodos):
    print("---------------------------")
    print("Año ",presupuesto["Periodo"][t])
    inv_periodo = 0
    for k in range(num_inversiones):
        if x[k,t].value >= 0.9:
           print(inversiones["Inversion"][k],"\t:",inversiones["Valor"][k]) 
           inv_periodo +=  inversiones["Valor"][k]
    print("Total     :\t",inv_periodo)
    print("Disponible:\t",presupuesto["Valor"][t])
