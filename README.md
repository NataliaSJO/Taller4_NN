# Taller 4 MIAX — Redes Neuronales para Forecasting

Forecasting de precios de cierre del S&P 500 (23 activos) usando redes neuronales.  
Módulo B3-T4 | Fecha de entrega: 21 de mayo de 2026

---

## Estructura del repositorio

```
Taller4_NN/
├── utilidades/              Código compartido (importado por todos los cuadernos)
│   ├── carga_datos.py       Descarga de datos, ventanas temporales, splits
│   ├── modelos.py           Factory functions: Dense, LSTM, Conv1D, Mixto
│   ├── evaluacion.py        MAE, guardar/cargar resultados JSON
│   └── graficos.py          Convergencia, barras MAE, resumen, matriz
│
├── cuadernos/               16 cuadernos — uno por combinación de ventanas
│   ├── ent05_sal01.ipynb    entrada=5 días / salida=1 día
│   ├── ent05_sal05.ipynb
│   ├── ...                  (4 entradas × 4 salidas = 16 combinaciones)
│   └── ent90_sal90.ipynb
│
├── analisis/
│   ├── graficos_resumen.ipynb   4 gráficos resumen + matriz de competición
│   └── cartera.ipynb            Tarea 2: cartera con/sin predicciones (2025)
│
├── resultados/
│   └── metricas/            JSON generados al ejecutar cada cuaderno
│                            (ent05_sal01.json, ...)
│
└── doc/                     Enunciado y guía de entrenamiento
```

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install yfinance keras tensorflow scikit-learn pandas numpy matplotlib
```

### 2. Ejecutar los cuadernos de experimentos

Abrir cualquier cuaderno de `cuadernos/` y ejecutar todas las celdas.  
La primera ejecución descarga los datos via yfinance y los guarda en  
`resultados/retornos_cache.csv`. Las siguientes ejecuciones usan el caché.

Cada cuaderno genera automáticamente su fichero JSON en `resultados/metricas/`.

### 3. Generar los gráficos resumen y la matriz de competición

Una vez ejecutados los 16 cuadernos, abrir `analisis/graficos_resumen.ipynb`.

### 4. Análisis de cartera (Tarea 2)

Abrir `analisis/cartera.ipynb`. Requiere que los cuadernos con `sal90` estén ejecutados.

---

## Modelos implementados

| Modelo      | Descripción                                     |
|-------------|--------------------------------------------------|
| Lineal      | Regresión lineal (sklearn) — baseline            |
| BuyAndHold  | Predice siempre la media del entrenamiento        |
| Dense       | MLP con entrada aplanada (256 → 128 → salida)    |
| LSTM        | Red recurrente con 64 unidades                   |
| Conv1D      | Convolucional 1D + GlobalAvgPool + Dense         |
| Mixto       | Conv1D → LSTM → Dense                           |

---

## Combinaciones de ventanas con datos de test
La siguiente tabla muestra el menor MAE obtenido en test para cada combinación de ventana de entrada y ventana de salida en la fase de competición.

| Ventana salida \ Ventana entradas | 1d | 5d | 30d | 90d |
|---|---|---|---|---|
| **5d**  | 0.012243 | 0.005581 | 0.002319 | 0.001263 |
| **10d** | 0.012244 | 0.005580 | 0.002319 | 0.001262 |
| **30d** | 0.012251 | 0.005578 | 0.002319 | 0.001264 |
| **90d** | 0.012271 | 0.005589 | 0.002320 | 0.001268 |

## Combinaciones de ventanas con datos de investigación
La siguiente tabla muestra el MAE obtenido en test para cada combinación de ventanas tras aplicar el preprocesado de la fase de investigación.

| Ventana salidas \ Ventana entradas | 1d | 5d | 30d | 90d |
|---|---|---|---|---|
| **5d**  | 0.715953 | 0.313762 | 0.105664 | 0.052030 |
| **10d** | 0.715963 | 0.290754 | 0.105601 | 0.052185 |
| **30d** | 0.715437 | 0.292693 | 0.107512 | 0.052152 |
| **90d** | 0.715383 | 0.291364 | 0.105679 | 0.052584 |
