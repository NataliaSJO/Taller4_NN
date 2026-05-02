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

## Combinaciones de ventanas

| Ventana entrada \ Ventana salida | 1d | 5d | 30d | 90d |
|---|---|---|---|---|
| **5d**  | ent05_sal01 | ent05_sal05 | ent05_sal30 | ent05_sal90 |
| **10d** | ent10_sal01 | ent10_sal05 | ent10_sal30 | ent10_sal90 |
| **30d** | ent30_sal01 | ent30_sal05 | ent30_sal30 | ent30_sal90 |
| **90d** | ent90_sal01 | ent90_sal05 | ent90_sal30 | ent90_sal90 |
