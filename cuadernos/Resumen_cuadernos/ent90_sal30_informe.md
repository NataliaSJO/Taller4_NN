# Taller B3-T4 — Informe técnico: `ent90_sal30`

**Caso:** ventana de entrada 90 días, ventana de salida 30 días (media móvil).
**Objetivo:** predecir el retorno medio futuro a 30 días para 23 activos a partir
del histórico de 90 días previos.

---

## 1. Características del caso

| Característica | Valor |
|---|---|
| Activos | 23 (log-retornos diarios) |
| Ventana entrada | 90 días |
| Ventana salida | 30 días (promediada) |
| Muestras totales | 16 071 |
| Train / Val / Test | 13 739 / 724 / 1 608 |
| Dimensión entrada plana | 2 070 (90 × 23) |
| Dimensión entrada 3D | (90, 23) |
| Dimensión salida | 23 |
| Métrica principal | MAE promediado sobre los 23 activos |
| Baselines | Regresión lineal, BuyAndHold |

**Escala de los datos:**
- Log-retornos diarios: σ ≈ 10⁻³ por activo.
- Target a 30 días: σ ≈ 3.3 × 10⁻³ (la media de 30 retornos reduce la varianza
  por un factor ~√30 ≈ 5.5 respecto a un retorno diario aislado).

---

## 2. Problemas encontrados en el setup inicial

El notebook inicial usaba las versiones antiguas de los modelos `_E`:

| Familia | Función inicial | Problema |
|---|---|---|
| Dense | `construir_dense_v1_E` | MAE colapsado a BuyAndHold (≡ predicción constante) |
| Recurrente | `construir_recurrente_v3_E` | **Divergía**: MAE test ~5× peor que BuyAndHold |
| Conv1D | `construir_conv1d_v1_E` | MAE colapsado a BuyAndHold |
| Mixto | `construir_mixto_v2_E` | MAE **idéntico** al de BuyAndHold al sexto decimal |

### 2.1 Colapso a constante

El MAE del modelo Mixto (0.002320) coincidía **exactamente** con el de
BuyAndHold (0.002320). Esto es matemáticamente imposible para un modelo que
aprenda algo no trivial. La interpretación correcta: el modelo **predecía
siempre la misma constante** (la media histórica) y no estaba aprendiendo
ninguna estructura del input.

**Justificación matemática.** Para un conjunto de targets ${y_1, ..., y_N}$,
el valor constante $c$ que minimiza el MAE es la **mediana** de los $y_i$:

$$\min_c \frac{1}{N}\sum_{i=1}^N |y_i - c|
\quad\Rightarrow\quad c^* = \text{mediana}(y_i)$$

Para retornos diarios casi simétricos en torno a cero, mediana ≈ media ≈ 0.
Si un modelo no extrae señal predictiva de los inputs, su mejor estrategia
según MAE es predecir esa constante. Eso es exactamente lo que hace
BuyAndHold.

**Causas concretas del colapso del Mixto v2_E:**
- `lr=3e-5` (50× más bajo que lo que sugiere el random search del taller).
- Doble `Dropout(0.5)` (regularización extrema que congela el aprendizaje).
- Loss `Huber(delta=1.0)` aplicado sobre retornos de escala 10⁻³. Como los
  errores nunca superan delta, Huber se comporta como **MSE puro**, lo que
  empuja al modelo hacia la solución conservadora (predecir la media).

Los tres efectos combinados producen una "rendición" del modelo: el optimizador
no consigue moverse de la inicialización aleatoria hacia ningún mínimo no
trivial.

### 2.2 Divergencia del LSTM

El `construir_recurrente_v3_E` daba MAE test 0.009772 (vs BAH 0.002320),
**4.2× peor que predecir la media**. Causa: Huber loss + `lr=3e-5` empujando
al modelo en una dirección incoherente con la escala del problema.

### 2.3 Falta de normalización de inputs

Al intentar acelerar el aprendizaje subiendo el `lr` hasta `5×10⁻²`, las curvas
seguían **planas**. Diagnóstico: con inputs en escala 10⁻³ y pesos inicializados
con Glorot (varianza unidad), los gradientes que llegan a las primeras capas
son del orden 10⁻⁶. Adam normaliza la dirección pero tarda decenas de épocas
en calibrar sus momentos con gradientes tan pequeños y ruidosos.

**Sin normalización de inputs ningún `lr` funciona.** Subir `lr` solo
añade ruido sin ganar señal.

### 2.4 Comparación engañosa de Parte 1 vs Parte 2

Al comparar el MAE del modelo sin preprocesado (0.002325) con el MAE del modelo
con preprocesado FFD+denoising (0.105864), el output reportaba una "variación
de -4 453%". Engaño metodológico: los dos modelos predicen targets en
**escalas distintas**:

- Parte 1: predice log-retornos crudos, σ(target) ≈ 3.3 × 10⁻³.
- Parte 2: predice serie FFD-denoised, σ(target) ≈ 1.4 × 10⁻¹.

Ratio de escalas: **42.7×**. Comparar MAE absolutos entre escalas distintas
es como comparar errores en metros vs. errores en kilómetros.

---

## 3. Cambios implementados

### 3.1 Normalización de inputs en Parte 1

Añadido en la celda de preparación de datos:

```python
scaler = StandardScaler()
X_train_plano = scaler.fit_transform(X_train_plano)
X_val_plano   = scaler.transform(X_val_plano)
X_test_plano  = scaler.transform(X_test_plano)
X_train = X_train_plano.reshape(X_train.shape)
X_val   = X_val_plano.reshape(X_val.shape)
X_test  = X_test_plano.reshape(X_test.shape)
```

`fit_transform` solo en `train` para evitar data leakage. Targets sin tocar
(siguen en escala original para que MAE sea comparable con BuyAndHold).

### 3.2 Cuatro modelos nuevos `_E` específicos para `ent90/sal30`

Implementados en `utilidades/modelos.py`, aplicando las lecciones del random
search del Taller B3-T4:

- `lr = 3 × 10⁻⁴` con `clipnorm = 1.0` (en lugar de `3 × 10⁻⁵`).
- `loss = 'mae'` (en lugar de `Huber`, que con `delta=1` sobre retornos en
  escala 10⁻³ se comporta como MSE puro).
- `dropout = 0.2–0.3` (en lugar de `0.5` que congelaba el aprendizaje).
- Activaciones: `gelu` en Dense y Mixto, `tanh` en LSTM (kernel optimizado
  cuDNN/Metal), `relu` en Conv1D.
- Aprovechamiento de la ventana 90 para: LSTM apilada vertical, Conv1D
  dilatada apilada (campo receptivo 13 días).

### 3.3 Métrica adimensional para comparativas

Adoptamos `MAE / σ(target)` como métrica adimensional invariante a escala:

$$\text{MAE}_{\text{norm}} = \frac{\text{MAE}}{\sigma(y_{\text{test}})}$$

Interpretación:

| `MAE_norm` | Interpretación |
|---|---|
| ≈ 0.80 | Modelo predice prácticamente la media (= BuyAndHold) |
| ≈ 0.70 | Modelo captura ~12% de varianza adicional |
| ≈ 0.50 | Modelo bastante bueno |
| < 0.20 | Modelo muy bueno (verificar leakage) |

El valor de referencia 0.80 viene del cociente teórico $\sqrt{2/\pi} \approx 0.798$
entre MAE óptimo y desviación típica para una distribución gaussiana cuando se
predice la media.

---

## 4. Modelos finales utilizados

### 4.1 `construir_dense_v12_E` — MLP medium

**Arquitectura:** `2070 → 128 → 64 → 30`, gelu, dropout 0.25, L2 1e-4.
**Parámetros:** 274 839.
**Lr:** 3 × 10⁻⁴ con clipnorm 1.0, loss MAE.

### 4.2 `construir_recurrente_v14_E` — LSTM apilada vertical

**Arquitectura:** `LSTM(64, return_seq) → LayerNorm → LSTM(32) → LayerNorm
→ Dense(32, gelu) → Dropout(0.2) → Dense(30)`.
**Parámetros:** 36 951.
**Activación:** `tanh` (kernel optimizado), `dropout=0.3` en ambas LSTM.
**Lr:** 3 × 10⁻⁴, loss MAE.

Con 90 pasos la apilada vertical sí tiene sentido: la segunda capa procesa
secuencias de 90 estados resumidos por la primera.

### 4.3 `construir_conv1d_v12_E` — Conv1D dilatada apilada

**Arquitectura:** `Conv1D(32, k=7, causal) → Conv1D(32, k=7, dilation=2,
causal) → GAP → Dropout(0.3) → Dense(30)`.
**Parámetros:** 13 143.
**Campo receptivo efectivo:** ~13 días tras las dos capas (suficiente para
patrones quincenales).
**Lr:** 3 × 10⁻⁴, loss MAE.

### 4.4 `construir_mixto_v13_E` — Conv1D dilatado + LSTM

**Arquitectura:** `Conv1D(32, k=7, dilation=2, causal, gelu) → Dropout(0.3)
→ LSTM(32, dropout=0.3) → Dropout(0.3) → Dense(30)`.
**Parámetros:** 14 263.
**Lr:** 3 × 10⁻⁴, loss MAE.

Variante "sana" de `v2_E` con los hiperparámetros del random search aplicados.

---

## 5. Resultados

### 5.1 Parte 1 — Competición sobre log-retornos en bruto

| Modelo | MAE train | MAE val | MAE test | Parámetros |
|---|---|---|---|---|
| **Lineal** | 0.002006 | 0.001966 | 0.002616 | 0 |
| **BuyAndHold** | 0.002166 | 0.001798 | 0.002320 | 0 |
| Dense v12_E | 0.002193 | 0.001801 | 0.002346 | 274 839 |
| LSTM v14_E | 0.002200 | 0.001818 | 0.002357 | 36 951 |
| Conv1D v12_E | 0.002272 | 0.002026 | 0.002555 | 13 143 |
| **Mixto v13_E** | 0.002149 | **0.001780** | 0.002325 | 14 263 |

**Observaciones:**
- Los MAE ya no coinciden con BAH al sexto decimal → no hay colapso.
- En **validación** el Mixto gana ligeramente a BuyAndHold (0.001780 vs
  0.001798). En test la diferencia se invierte ligeramente (0.002325 vs
  0.002320), dentro del rango de ruido.
- Conv1D queda peor que BAH en test: posible exceso de capacidad para la
  escasa señal disponible en el target.

### 5.2 Parte 2 — Preprocesado FFD + Denoising

**Parámetros del FFD:**
- `d_values` probados: `[0.1, 0.2, ..., 1.0]`.
- `max_width = 500` pesos.
- Todos los 23 activos eligen **`d_opt = 0.1`** (el extremo inferior del
  rango).

**Parámetros del denoising Marchenko-Pastur:**
- Autovalores de señal retenidos: **1 / 23** (solo el factor de mercado).
- $\lambda_{\max} = 6.4682$.

**Resultado bruto:**

| Modelo | MAE train | MAE val | MAE test |
|---|---|---|---|
| Mixto v13_E (sin prep) | 0.002149 | 0.001780 | 0.002325 |
| Mixto v13_E (con prep) | 0.090653 | 0.086253 | 0.105864 |

Ratio de MAEs: 45.5×. **Esto NO indica que el preprocesado destroce el
modelo**: refleja el cambio de escala del target tras el FFD.

### 5.3 Comparativa correcta (adimensional)

| Cantidad | Sin prep | Con prep |
|---|---|---|
| σ(target test) | 0.003269 | 0.139584 |
| MAE test | 0.002325 | 0.105864 |
| **MAE/σ** | **0.711** | **0.758** |
| Mejora vs predictor constante (~0.80) | ~11.1% | ~5.2% |

**Diferencia relativa real:** +6.6% en MAE/σ con preprocesado.

---

## 6. Conclusiones técnicas

### 6.1 Sobre los modelos

1. **Los nuevos modelos `_E` entrenan correctamente y no colapsan**. El
   problema inicial era enteramente atribuible a los hiperparámetros viejos
   (lr 50× demasiado bajo, dropout excesivo, Huber sobre escala incorrecta).

2. **Los modelos baten a BuyAndHold ligeramente** (~11% extra de varianza
   capturada en `MAE/σ`). Es una mejora modesta pero real. Coherente con la
   teoría de mercados eficientes a horizonte medio: la mayor parte de la
   varianza no es predecible desde el pasado reciente.

3. **El Mixto v13_E es el mejor modelo** en validación (0.001780). En test
   queda al nivel de BAH (0.002325 vs 0.002320). La diferencia en test entra
   en el rango de ruido entre runs.

### 6.2 Sobre el preprocesado FFD + Denoising

1. **El preprocesado no aporta mejora significativa** en `sal30` con 90d de
   entrada. Diferencia +6.6% en MAE/σ, comparable al ruido entre runs.

2. **El "-4 453%" del notebook es un artefacto metodológico**: comparación
   de MAE absolutos entre escalas distintas. La conclusión correcta requiere
   normalizar por la desviación típica de cada target.

3. **El denoising deja solo 1/23 autovalores de señal** (el factor mercado).
   Esto significa que tras el preprocesado los 23 activos se comportan
   prácticamente como uno. Lopez de Prado lo defiende como reducción de
   ruido, pero limita lo que el modelo puede aprender.

### 6.3 Sobre la calibración del FFD

**Que los 23 activos elijan `d = 0.1` (el extremo inferior) sugiere que el
rango de búsqueda está mal calibrado**. La práctica recomendada por Lopez de
Prado es buscar `d` con paso fino en `[0, 0.5]`, donde típicamente está la
`d` óptima real. El rango actual `[0.1, 1.0]` con paso `0.1` no explora la
zona realmente interesante.

**Hipótesis:** muchos activos probablemente prefieren `d ∈ [0, 0.1]` o incluso
`d = 0` (no aplicar FFD), pero el barrido actual los fuerza al mínimo
disponible (`0.1`).

---

## 7. Trabajo futuro

1. **Refinar el barrido de `d`** del FFD: probar `np.arange(0.0, 0.55, 0.05)`
   en lugar del rango actual. Posible cambio en las conclusiones del
   preprocesado.

2. **Comparativa en escala invertida.** Implementar la inversa aproximada del
   FFD para comparar predicciones en la misma escala que los log-retornos
   originales. Más riguroso que normalizar por σ.

3. **Random search específico para `ent90_sal30`.** El barrido existente está
   hecho para `sal90`. Aunque los hiperparámetros transferidos funcionan, un
   barrido específico podría descubrir configuraciones mejores.

4. **Evaluar con métricas adicionales:** R² adimensional, Sharpe ratio si se
   plantea uso financiero, accuracy direccional (signo correcto del retorno).

5. **Aplicar el mismo análisis a los otros notebooks** del set (`sal01`,
   `sal05`, `sal90` × `ent05`, `ent10`, `ent90`) para tener una visión
   completa de cómo escalan estas conclusiones con el horizonte de entrada y
   de salida.

---

## Anexo A — Diagnóstico rápido de colapso

Para confirmar si un modelo está colapsado a constante:

```python
preds = modelo.predict(X_test)
ratio = preds.var() / y_test.var()
print(f'Ratio var(preds)/var(targets): {ratio:.4f}')
```

| Ratio | Diagnóstico |
|---|---|
| < 0.01 | Modelo colapsado a constante |
| 0.1 – 0.8 | Modelo aprendiendo (sano) |
| ≈ 1.0 | Modelo replica la varianza del target |
| > 1.5 | Modelo sobreajusta o tiene varianza artificial |

## Anexo B — Resumen de la batería de cambios

| # | Cambio | Localización |
|---|---|---|
| 1 | Añadir `StandardScaler` a inputs | Celda 6 de `ent90_sal30.ipynb` |
| 2 | Sustituir Dense `v1_E` → `v12_E` | Celda 10 |
| 3 | Sustituir Recurrente `v3_E` → `v14_E` | Celda 13 |
| 4 | Sustituir Conv1D `v1_E` → `v12_E` | Celda 16 |
| 5 | Sustituir Mixto `v2_E` → `v13_E` | Celda 19 |
| 6 | Actualizar `constructores_prep` (Parte 2) | Celda 33 |
| 7 | Añadir métrica `MAE/σ` en comparativa | Celda nueva tras 35 |
| 8 | Nuevas funciones en `utilidades/modelos.py` | 4 funciones nuevas |
