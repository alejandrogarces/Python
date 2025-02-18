**Determinación de un portafolio de inversiones para un operador de red**

___

En este documento se determina un portafolio óptimo de inversiones usando un modelo lineal entero.
Las entradas son las inversiones y su correspondiente beneficio anual, así como el presupuesto por año.

___
*Variables de decisión*:

$x_{it}$: variable binaria que indica si se hace la inversión $i$ en el periodo $t$

___
*Parametros*:

+ $c_i$: costo de la inversión $i$ en miles de millones
+ $b_i$: beneficio anual de la inversión $i$ en miles de millones
+ $p_t$: presupuesto disponible en el periodo $t$

___
Función objetivo

$$ \min \sum_t (f_t-g_t)$$

en donde $f_t$ y $g_t$ son:
+ Inversiones totales:

$$f = \sum_{i} c_ix_{it} $$

+ beneficio total:

$$ g = \sum_{i} b_ix_{it}$$
___
Restricciones

+ Solo una inversión en un tiempo:

$$ \sum_t c_{i}x_{it} = 1, \; \forall i$$

+ inversión en cada periodo menor al presupuesto disponible

$$ f_t \leq p_t, \; \forall t$$

