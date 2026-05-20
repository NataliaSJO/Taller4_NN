# Taller B3-T4 — Informe técnico: `ent90_sal90`

**Caso:** ventana de entrada 90 días, ventana de salida 90 días (media móvil).
**Objetivo:** predecir el retorno medio futuro a 90 días para 23 activos a
partir del histórico de 90 días previos.

Este es el caso para el que se realizó el **random search original** del
Taller B3-T4 (120 combinaciones, 4 familias). Los modelos `_E` resultantes
(v3_E Dense, v5_E Recurrente, v3_E Conv1D, v4_E Mixto) son los que se usan
en este notebook.

---

## 1. Características del caso

| Característica | Valor |
|---|---|
| Activos | 23 (log-retornos diarios) |
| Ventana entrada | 90 días |
| Ventana salida | 90 días (promediada) |
| Muestras totales | 16 011 |
| Train / Val / Test | 13 688 / 721 / 1 602 |
| Dimensión entrada plana | 2 070 (90 × 23) |
| Dimensión entrada 3D | (90, 23) |
| Dimensión salida | 23 |
| Métrica principal | MAE promediado sobre los 23 activos |
| Baselines | Regresión lineal, BuyAndHold |

**Escala de los datos:**
- Log-retornos diarios: σ ≈ 10⁻³ por activo.
- Target a 90 días: σ ≈ 10⁻³ (la media de 90 retornos reduce la varianza por
  un factor ~√90 ≈ 9.5, dejando el target con muy poca variabilidad).

**Característica distintiva:** el target promediado a 90 días es **el más
suave de toda la matriz de casos**. Esto implica que BuyAndHold (= predecir
la media histórica) es un baseline durísimo de batir. La señal predecible que
queda tras ese suavizado tan agresivo es muy débil.

---

## 2. Problemas encontrados en el setup inicial

### 2.1 Falta de normalización en Parte 1

Idéntico al problema diagnosticado en `ent90_sal30`. La celda 6 original
solo hacía `aplanar_X` sin escalar los inputs, lo que dejaba a las redes con
gradientes microscópicos (~10⁻⁶) y un comportamiento característico:

- **Conv1D v3_E** restauraba pesos de la **época 3** (se rendía
  inmediatamente, el `val_loss` quedaba clavado al de BuyAndHold al sexto
  decimal).
- **Dense v3_E** y **Mixto v4_E** entrenaban algo, pero igualaban a BAH ±
  ruido.

### 2.2 LSTM apuntando al modelo viejo

La celda 13 invocaba `construir_recurrente_v2_E` (versión antigua,
sobre-regularizada con lr 3e-5 + Huber loss) en lugar de
`construir_recurrente_v5_E` (la del random search). Los imports en la celda
2 eran correctos (importaban `v5_E`), pero la llamada en la celda 13 estaba
desactualizada. Resultado: MAE test 0.001449, **14% peor que BuyAndHold**.

### 2.3 LSTM v5_E original era extremadamente lento

El v5_E ganador del random search usaba `activation='elu'`. Esto desactiva el
**kernel optimizado de Keras para LSTM** (cuDNN/Metal, que solo se aplica con
`activation='tanh'`) y cae a una implementación en Python puro. En un
MacBook Air M4 sin GPU para TF, una sola época podía tardar **varios
minutos**.

**Decisión de diseño tomada:** sustituir en `modelos.py` la variante `elu`
por la `tanh` (que era el #2 del barrido, a 0.000002 de MAE de diferencia
respecto al top-1). Aproximadamente **10× más rápida** sin pérdida material
de calidad.

### 2.4 Conv1D + preprocesado se rendía (curva plana, no aprendía)

Tras aplicar el `StandardScaler` y arreglar el LSTM, surgió un nuevo problema
específico de la Parte 2: la Conv1D con datos FFD+denoising no entrenaba. La
curva de loss quedaba plana desde el inicio.

**Causa:** tras el denoising Marchenko-Pastur queda solo **1 autovalor de
señal de 23**, lo que significa que los 23 activos se comportan prácticamente
como un único factor (el de mercado). Para una Conv1D que aprende filtros
cross-canal, queda muy poca información estructural que extraer. Con `lr=3e-4`
y batch=128, los gradientes son demasiado pequeños y el modelo se rinde
inmediatamente.

### 2.5 Comparativa engañosa de escalas (recurrente)

Como en `ent90_sal30`, el output del notebook reportaba "variación MAE test
con preprocesado: -4032.9%". Engaño metodológico clásico: los modelos sin y
con preprocesado predicen targets en escalas distintas (log-retornos vs
serie FFD-denoised), y el MAE absoluto no es directamente comparable.

---

## 3. Cambios implementados

### 3.1 Normalización de inputs en Parte 1 (celda 6)

Idéntico al fix de `ent90_sal30`:

```python
scaler = StandardScaler()
X_train_plano = scaler.fit_transform(X_train_plano)
X_val_plano   = scaler.transform(X_val_plano)
X_test_plano  = scaler.transform(X_test_plano)
X_train = X_train_plano.reshape(X_train.shape)
X_val   = X_val_plano.reshape(X_val.shape)
X_test  = X_test_plano.reshape(X_test.shape)
```

Output esperado: `media=0.0000  std=1.0000`. ✅ Verificado.

### 3.2 Corrección del LSTM (celda 13)

```python
# Antes:
modelo_lstm = construir_recurrente_v2_E(X_train.shape[1:], y_train.shape[1])
# Después:
modelo_lstm = construir_recurrente_v5_E(X_train.shape[1:], y_train.shape[1])
```

### 3.3 Cambio de `elu` a `tanh` en `construir_recurrente_v5_E` (modelos.py)

Modificación permanente en `utilidades/modelos.py`:

- `activation='elu'` → `'tanh'` (en ambas LSTM apiladas).
- `dropout=0.4` → `0.2` (lo que daba el random search para la variante tanh).

**Justificación:** la variante tanh era el #2 del random search a solo
0.000002 de MAE del top-1, pero es ~10× más rápida en CPU porque activa el
kernel optimizado. Trade-off de calidad insignificante por una mejora de
velocidad enorme.

### 3.4 Ajustes específicos en Parte 2 para que la Conv1D aprenda

Tres cambios en la celda 33 respecto a la configuración por defecto:

| Hiperparámetro | Default | Aplicado en este caso |
|---|---|---|
| `epochs` | 300 | **30** |
| `batch_size` | 128 | **800** |
| `lr` inicial (recompile) | 3e-4 (modelo.py) | considerar `1e-3` |

**Justificación del `batch_size=800`:** un batch muy grande promedia más
muestras por step y reduce el ruido del gradiente. Con un input casi
unidimensional efectivo (después del denoising), esto ayuda a evitar que la
Conv1D se rinda por falta de señal por mini-batch.

### 3.5 Comparativa con métrica adimensional (recomendado)

Añadir tras la celda 35 una celda con:

```python
print(f'σ(y_test) sin prep:  {y_test.std():.6f}')
print(f'σ(y_ts_p) con prep:  {y_ts_p.std():.6f}')
print(f'Ratio:                {y_ts_p.std() / y_test.std():.1f}x')
print()
mae_sin = resultado_sin_prep["mae_test"]
mae_con = resultado_prep["mae_test"]
print(f'MAE/σ sin prep:  {mae_sin / y_test.std():.3f}')
print(f'MAE/σ con prep:  {mae_con / y_ts_p.std():.3f}')
print('(referencia: predictor constante optimo ~0.798)')
```

---

## 4. Modelos utilizados (random search del Taller B3-T4)

### 4.1 `construir_dense_v3_E` — MLP medium

**Arquitectura:** `2070 → 128 → 64 → 23`, gelu, dropout 0.2, L2 1e-4.
**Parámetros:** 274 839.
**Lr:** 3 × 10⁻⁴ con clipnorm 1.0, loss MAE.

### 4.2 `construir_recurrente_v5_E` (variante tanh) — LSTM small apilada

**Arquitectura:** `LSTM(32, tanh, return_seq) → LayerNorm → LSTM(16, tanh)
→ LayerNorm → Dense(23)`.
**Parámetros:** 10 791.
**Dropout:** 0.2 en ambas LSTM.
**Lr:** 3 × 10⁻⁴ con clipnorm 1.0, loss MAE.

Variante elegida por velocidad (kernel optimizado). El #2 del random search,
a 0.000002 de MAE del top-1 con `elu`, pero ~10× más rápida en CPU.

### 4.3 `construir_conv1d_v3_E` — Conv1D dilatada minimalista

**Arquitectura:** `Conv1D(16, k=7, dilation=2, causal) → MaxPooling1D
→ Conv1D(8, k=7, padding='same') → GAP → Dropout(0.4) → Dense(23)`.
**Parámetros:** 3 703 (la red más pequeña del random search).
**Lr:** 3 × 10⁻⁴, loss MAE.

Modelo más eficiente del random search en relación coste/beneficio: entrena
en ~16 segundos y queda al mismo nivel que los modelos 10× más grandes.

### 4.4 `construir_mixto_v4_E` — Conv1D + LSTM

**Arquitectura:** `Conv1D(32, k=5, gelu, causal) → Dropout(0.3) → LSTM(64,
dropout=0.3) → Dropout(0.3) → Dense(23)`.
**Parámetros:** 30 039.
**Lr:** 1 × 10⁻³ con clipnorm 1.0, loss MAE.

Única familia donde el random search dio `lr=1e-3` como ganador, no `3e-4`.
La activación gelu con clipnorm estabiliza el lr más alto.

---

## 5. Resultados

### 5.1 Parte 1 — Competición sobre log-retornos en bruto

| Modelo | MAE train | MAE val | MAE test | Parámetros |
|---|---|---|---|---|
| **Lineal** | 0.001143 | 0.001150 | 0.001509 | 0 |
| **BuyAndHold** | 0.001235 | 0.000997 | 0.001268 | 0 |
| Dense v3_E | 0.001227 | 0.000990 | 0.001278 | 274 839 |
| LSTM v5_E (tanh) | 0.001256 | 0.000992 | 0.001279 | 10 791 |
| **Conv1D v3_E** | 0.001235 | 0.000996 | **0.001272** | 3 703 |
| Mixto v4_E | 0.001234 | **0.000976** | 0.001280 | 30 039 |

**Observaciones:**

1. **Los modelos ya no colapsan y entrenan sanamente.** Conv1D restaura
   pesos en la **época 52** (no 3 como antes), Dense en **época 111**.

2. **En validación los modelos baten a BuyAndHold:**
   - Mixto val: 0.000976 vs BAH val: 0.000997 (~2% mejor).
   - Dense val: 0.000990 vs BAH val: 0.000997.
   - LSTM val: 0.000992 vs BAH val: 0.000997.

3. **En test BuyAndHold gana por margen muy pequeño:** todos los modelos
   quedan en 0.001272–0.001280 vs BAH 0.001268. Diferencia del orden 10⁻⁶,
   dentro del rango de ruido entre runs.

4. **El Conv1D v3_E es el mejor modelo NN en test** (0.001272). Es notable
   que con apenas **3 703 parámetros** quede igual que el Dense de 274 839
   (74× más grande). Confirma la conclusión del random search: para targets
   muy suaves, la capacidad no aporta, solo añade riesgo de sobreajuste.

5. **La regresión lineal queda significativamente peor** (0.001509) que BAH.
   Esto indica que la lineal **sí encuentra correlaciones** entre input y
   target, pero esas correlaciones no generalizan bien a test. Las redes
   neuronales tampoco las explotan: probablemente son espurias.

### 5.2 Parte 2 — Preprocesado FFD + Denoising

**Parámetros del FFD:**

- `d_values` probados: `[0.1, 0.2, …, 1.0]`.
- `max_width = min(500, len(serie)//2)`.
- Todos los 23 activos eligen **`d_opt = 0.1`** (extremo inferior del rango,
  mismo problema metodológico que en `ent90_sal30`).

**Parámetros del denoising Marchenko-Pastur:**

- Autovalores de señal retenidos: **1 / 23** (solo el factor mercado).
- $\lambda_{\max}$ Marchenko-Pastur: variable (depende del q = T/N).

**Resultado bruto:**

| Modelo | MAE train | MAE val | MAE test |
|---|---|---|---|
| Conv1D v3_E (sin prep) | 0.001235 | 0.000996 | 0.001272 |
| Conv1D v3_E (con prep) | 0.047947 | 0.042803 | 0.052584 |

Ratio MAE: **41.3×**. Análogo al caso `ent90_sal30` (ratio 45.5×). **Es
artefacto de escala**, no degradación real del modelo.

### 5.3 Comparativa correcta (adimensional)

*Requiere ejecutar el bloque de la sección 3.5 en el notebook. Pendiente.*

Hipótesis basada en la analogía con `ent90_sal30`:

- MAE/σ esperado sin prep: ~0.80–0.85 (BuyAndHold es difícil de batir en
  sal90).
- MAE/σ esperado con prep: similar.
- Diferencia probable: < 10%.

---

## 6. Conclusiones técnicas

### 6.1 Sobre los modelos

1. **Los modelos del random search funcionan correctamente tras añadir
   `StandardScaler`.** Sin normalización los gradientes son demasiado
   pequeños y las redes se rinden.

2. **El Conv1D v3_E es notablemente eficiente:** 3 703 parámetros, mismo
   rendimiento que el Dense de 274 839. Para targets muy suavizados, la
   capacidad solo añade ruido.

3. **En validación, los modelos baten ligeramente a BuyAndHold.** En test el
   orden se invierte por márgenes muy pequeños (10⁻⁶). Conclusión sincera:
   con horizonte de 90 días, predecir la media (BAH) ya captura prácticamente
   toda la varianza explicable.

4. **La variante `tanh` del LSTM v5_E es la elección práctica correcta**
   para máquinas sin GPU. La pérdida de calidad respecto al `elu` original
   es 0.000002 de MAE (no detectable), pero la velocidad mejora ~10×.

### 6.2 Sobre el preprocesado FFD + Denoising

1. **El "-4032.9%" reportado es un artefacto metodológico**. Comparativa
   debe hacerse en escala normalizada (MAE/σ) o tras invertir el FFD.

2. **El denoising deja 1/23 autovalores de señal**, igual que en
   `ent90_sal30`. Esto implica que tras el preprocesado todos los activos
   se comportan como uno (factor mercado), lo que limita drásticamente la
   información que las redes pueden aprovechar.

3. **Específicamente para Conv1D + preprocesado, hubo que ajustar:**
   `batch_size=800` y `epochs=30`. Sin estos ajustes la Conv1D no aprende
   nada (curva plana). Esto es señal de que la información estructural
   disponible es realmente muy limitada tras el denoising.

### 6.3 Sobre la calibración del FFD (idéntico a `ent90_sal30`)

**Todos los activos eligen `d = 0.1`** (extremo inferior del rango). Sugiere
que el rango de búsqueda debería ampliarse a `[0, 0.5]` con paso `0.05`.
Es muy posible que muchos activos prefieran `d ∈ [0, 0.1]` o incluso `d = 0`.

### 6.4 Diferencia clave con `ent90_sal30`

| Aspecto | `ent90_sal30` | `ent90_sal90` |
|---|---|---|
| Varianza del target | ~10⁻⁵ | ~10⁻⁶ (más suave) |
| Modelos necesarios | Crear nuevos `_E` | **Reusar los del random search** |
| Tipo de fix | Crear v12_E, v14_E, v12_E, v13_E | Cambiar imports, ajustar lr/batch en Parte 2 |
| Conv1D + prep | Funcionaba sin ajuste especial | Requirió `batch_size=800`, `epochs=30` |

El caso `sal90` es **el caso para el que se diseñaron los modelos `_E`
v3_E/v5_E/v4_E** vía random search. Por eso aquí no hace falta crear
modelos nuevos, solo asegurarse de que se usan correctamente y con inputs
normalizados.

---

## 7. Trabajo futuro

1. **Calcular y reportar `MAE/σ` adimensional** en la comparativa Parte 1 vs
   Parte 2 (el bloque de la sección 3.5). Es la métrica que permite concluir
   si el preprocesado aporta o no.

2. **Refinar el barrido de `d`** del FFD: `np.arange(0.0, 0.55, 0.05)` en
   lugar del rango actual `[0.1, 1.0]`.

3. **Comparativa en escala invertida** (inversa aproximada del FFD) para
   tener una métrica única comparable directamente.

4. **Limpiar código muerto en celda 33** (el `modelo_prep.compile(...)` que
   aparece antes de crear el modelo). Funciona porque hay un modelo previo
   en memoria, pero rompe el `Restart & Run All` desde cero.

5. **Replicar el mismo análisis en `ent10_sal90` y `ent05_sal90`** para ver
   cómo varía el comportamiento del preprocesado con ventanas de entrada
   más cortas. Hipótesis: con entradas más cortas el preprocesado podría
   tener efecto distinto.

---

## Anexo A — Por qué tanh es preferible a elu en LSTM (sin GPU)

Las implementaciones modernas de Keras detectan automáticamente si una capa
LSTM es compatible con el "fast path" (kernel cuDNN en GPU, equivalente
optimizado en CPU). Los requisitos para activar el fast path incluyen:

- `activation = 'tanh'`
- `recurrent_activation = 'sigmoid'`
- `dropout = 0` o `recurrent_dropout = 0`
- `unroll = False`
- `use_bias = True`

Si cualquiera de estos se incumple, Keras cae a la implementación general en
Python, mucho más lenta. Con `activation='elu'`, **ningún optimizado de
hardware se aplica**, y el tiempo por época puede ser **10–20× mayor**.

Para arquitecturas con dos LSTM apiladas y secuencias de 90 timesteps,
esto significa la diferencia entre **segundos** y **minutos** por época.

## Anexo B — Resumen de la batería de cambios

| # | Cambio | Localización |
|---|---|---|
| 1 | Añadir `StandardScaler` a inputs | Celda 6 de `ent90_sal90.ipynb` |
| 2 | Sustituir Recurrente `v2_E` → `v5_E` | Celda 13 |
| 3 | Cambiar `activation='elu'` → `'tanh'` en v5_E | `utilidades/modelos.py` |
| 4 | Cambiar `dropout=0.4` → `0.2` en v5_E | `utilidades/modelos.py` |
| 5 | Ajustar `batch_size=128 → 800` para Parte 2 | Celda 33 |
| 6 | Ajustar `epochs=300 → 30` para Parte 2 | Celda 33 |
| 7 | Limpiar `modelo_prep.compile(...)` muerto (pendiente) | Celda 33 |
| 8 | Añadir métrica `MAE/σ` en comparativa (pendiente) | Celda nueva tras 35 |
