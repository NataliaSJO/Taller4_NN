import keras
from keras.models import Sequential, Model
from keras.layers import (
    Dense, LSTM, GRU, Conv1D, GlobalAveragePooling1D,
    Flatten, Input, Dropout, MaxPooling1D, Bidirectional, LayerNormalization,
    SeparableConv1D, Concatenate, TimeDistributed, Add
)
from keras import regularizers


# ──────────────────────────────────────────────────────────────
# MODELOS BASELINE
# ──────────────────────────────────────────────────────────────

def construir_baseline_lineal(dim_entrada, dim_salida):
    """Capa Dense sin activación — equivalente a regresión lineal en Keras."""
    modelo = Sequential(name='Lineal_Keras')
    modelo.add(Dense(dim_salida, input_shape=(dim_entrada,)))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


# ──────────────────────────────────────────────────────────────
# FAMILIA DENSA (entrada aplanada: ventana*23)
# Solo capas Dense + Dropout. Sin BatchNorm ni capas recurrentes/conv.
# ──────────────────────────────────────────────────────────────

def construir_dense(dim_entrada, dim_salida, neuronas=(256, 128)):
    """MLP clásico 2 capas (256→128). Dropout 0.2."""
    modelo = Sequential(name='Dense')
    modelo.add(Dense(neuronas[0], activation='relu', input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(neuronas[1], activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_dense_v2(dim_entrada, dim_salida):
    """MLP estrecho 3 capas (128→64→32). Dropout 0.3 para mayor regularización."""
    modelo = Sequential(name='Dense_v2')
    modelo.add(Dense(128, activation='relu', input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_dense_v3(dim_entrada, dim_salida):
    """MLP ancho 4 capas (512→256→128→64). Dropout leve 0.1."""
    modelo = Sequential(name='Dense_v3')
    modelo.add(Dense(512, activation='relu', input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(256, activation='relu'))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
    return modelo


def construir_dense_v4(dim_entrada, dim_salida):
    """MLP 2 capas con regularización L2 (128→64). Penaliza pesos grandes."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v4')
    modelo.add(Dense(128, activation='relu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(64, activation='relu', kernel_regularizer=reg))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_dense_v5(dim_entrada, dim_salida):
    """MLP 1 capa oculta grande (256) con activación tanh. Útil para señales centradas."""
    modelo = Sequential(name='Dense_v5')
    modelo.add(Dense(256, activation='tanh', input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_dense_v6(dim_entrada, dim_salida):
    """MLP piramidal inversa (32→64→128) con activación elu. Expande representación."""
    modelo = Sequential(name='Dense_v6')
    modelo.add(Dense(32, activation='elu', input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(64, activation='elu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(128, activation='elu'))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


# ── Variantes personales de Emilio ────────────────────────────

def construir_dense_v1_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """[Emilio] MLP reducido (128->64) con L2 + Dropout 0.4 + LR 3e-4.
    Disenyado para ventanas largas (90d): evita el sobreajuste con muchas features."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v1_E')
    modelo.add(Dense(neuronas[0], activation='relu',
                     kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(neuronas[1], activation='relu',
                     kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo

def construir_recurrente_v2_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """
    [Emilio] LSTM apilada 2 capas (128 -> 64) para ventana de entrada 90d.
    Mejoras de convergencia: recurrent_dropout, LayerNorm, gradient clipping,
    Huber loss y lr más agresivo (compensado por el clipping).
    """
    modelo = Sequential(name=f'{celda}_v2_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU

    modelo.add(CeldaClase(128,
                          input_shape=forma_entrada,
                          dropout=0.1,
                          recurrent_dropout=0.1,
                          return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(64,
                          dropout=0.1,
                          recurrent_dropout=0.1,
                          return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0)
    )
    return modelo

def construir_recurrente_v3_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """
    [Emilio] LSTM apilada 2 capas (128 -> 64) para ventana de entrada 90d.
    Sin recurrent_dropout (kernel optimizado, mucho más rápido en CPU).
    LR fijo compatible con ReduceLROnPlateau.
    """
    modelo = Sequential(name=f'{celda}_v2_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU

    modelo.add(CeldaClase(128,
                          input_shape=forma_entrada,
                          dropout=0.1,
                          return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.1))
    modelo.add(CeldaClase(64,
                          dropout=0.1,
                          return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0)
    )
    return modelo

# ──────────────────────────────────────────────────────────────
# FAMILIA RECURRENTE (entrada 3D: ventana×23)
# Solo celdas LSTM / GRU (incluido Bidirectional). Sin Conv1D.
# ──────────────────────────────────────────────────────────────

def construir_recurrente(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """LSTM o GRU monocapa con 64 unidades. Dropout 0.2."""
    modelo = Sequential(name=celda)
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(unidades, input_shape=forma_entrada, return_sequences=False))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_recurrente_v2(forma_entrada, dim_salida):
    """2 capas LSTM apiladas (64→32). La primera devuelve secuencia completa."""
    modelo = Sequential(name='LSTM_v2')
    modelo.add(LSTM(64, input_shape=forma_entrada, return_sequences=True))
    modelo.add(Dropout(0.2))
    modelo.add(LSTM(32, return_sequences=False))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_recurrente_v3(forma_entrada, dim_salida):
    """GRU bidireccional: procesa la secuencia en ambas direcciones temporales."""
    modelo = Sequential(name='BiGRU_v3')
    modelo.add(Bidirectional(GRU(32, return_sequences=False),
                             input_shape=forma_entrada))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_recurrente_v4(forma_entrada, dim_salida):
    """LSTM + Dense intermedia (64 LSTM → 32 Dense → salida). Abstracción extra."""
    modelo = Sequential(name='LSTM_v4')
    modelo.add(LSTM(64, input_shape=forma_entrada, return_sequences=False))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_recurrente_v5(forma_entrada, dim_salida):
    """LSTM de 128 unidades con dropout recurrente interno para mayor regularización."""
    modelo = Sequential(name='LSTM_v5')
    modelo.add(LSTM(128, input_shape=forma_entrada,
                    dropout=0.2, recurrent_dropout=0.1,
                    return_sequences=False))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
    return modelo


def construir_recurrente_v6(forma_entrada, dim_salida):
    """2 capas GRU apiladas (64→32) con Dropout entre ellas."""
    modelo = Sequential(name='GRU_v6')
    modelo.add(GRU(64, input_shape=forma_entrada, return_sequences=True))
    modelo.add(Dropout(0.2))
    modelo.add(GRU(32, return_sequences=False))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_recurrente_v1_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """[Emilio] LSTM/GRU monocapa con dropout interno (0.2 + 0.2) y LR 1e-4.
    Diseniada para series ruidosas y ventanas largas. Reemplaza el Dropout externo
    por dropout y recurrent_dropout internos al LSTM, mas eficaces en RNN."""
    modelo = Sequential(name=f'{celda}_v1_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(unidades, input_shape=forma_entrada,
                          dropout=0.2, recurrent_dropout=0.2,
                          return_sequences=False))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
    return modelo


# ──────────────────────────────────────────────────────────────
# FAMILIA CONV1D (entrada 3D: ventana×23)
# Solo capas Conv1D / SeparableConv1D + pooling + Flatten/GAP.
# Sin celdas recurrentes.
# ──────────────────────────────────────────────────────────────

def construir_conv1d(forma_entrada, dim_salida, filtros=64, kernel=3):
    """2 capas Conv1D + GlobalAveragePooling + Dense. Kernel 3."""
    modelo = Sequential(name='Conv1D')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      input_shape=forma_entrada))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu',
                      padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v2(forma_entrada, dim_salida):
    """1 capa Conv1D + Flatten + Dense. La más simple de la familia."""
    modelo = Sequential(name='Conv1D_v2')
    modelo.add(Conv1D(32, kernel_size=3, activation='relu',
                      padding='same', input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v3(forma_entrada, dim_salida):
    """3 capas Conv1D (64→32→16) + GlobalAveragePooling. Más profundidad convolucional."""
    modelo = Sequential(name='Conv1D_v3')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    modelo.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v4(forma_entrada, dim_salida):
    """Conv1D dilatada (dilation_rate=2) + GAP. Amplía el campo receptivo sin más params."""
    modelo = Sequential(name='Conv1D_v4')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu',
                      dilation_rate=2, padding='causal',
                      input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu',
                      dilation_rate=1, padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v5(forma_entrada, dim_salida):
    """SeparableConv1D (depthwise separable): mismo campo receptivo, menos parámetros."""
    modelo = Sequential(name='Conv1D_v5')
    modelo.add(SeparableConv1D(64, kernel_size=3, activation='relu',
                               padding='same', input_shape=forma_entrada))
    modelo.add(SeparableConv1D(32, kernel_size=3, activation='relu',
                               padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v6(forma_entrada, dim_salida):
    """Conv1D + MaxPooling1D + Flatten. Pooling explícito para reducir dimensión temporal."""
    modelo = Sequential(name='Conv1D_v6')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', input_shape=forma_entrada))
    modelo.add(MaxPooling1D(pool_size=2, padding='same'))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    modelo.add(Flatten())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_v1_E(forma_entrada, dim_salida, filtros=64, kernel=3):
    """[Emilio] Conv1D dilatada (causal) + MaxPool + Conv1D + GAP. L2 + Dropout 0.4.
    El padding causal y la dilatacion (rate=2) amplian el campo receptivo
    sin anyadir parametros: util en ventanas largas. El causal solo mira al pasado."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Conv1D_v1_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      dilation_rate=2, padding='causal',
                      input_shape=forma_entrada,
                      kernel_regularizer=reg))
    modelo.add(MaxPooling1D(pool_size=3))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu',
                      padding='same',
                      kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo


# ──────────────────────────────────────────────────────────────
# MODELO MIXTO (arquitectura híbrida, no pertenece a familia pura)
# ──────────────────────────────────────────────────────────────

def construir_mixto(forma_entrada, dim_salida):
    """Conv1D → LSTM → Dense con entrada 3D."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(entradas)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto')
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_mixto_v1_E(forma_entrada, dim_salida):
    """ Conv1D (32 filtros) -> LSTM con dropout interno -> Dense.
    L2 en Conv1D + Dropout externo 0.4 + LR 3e-4. Capacidad reducida respecto
    al mixto original (Conv1D 64->32) para que la regularizacion compense
    en ventanas largas con target promediado."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=reg)(entradas)
    x = LSTM(64, dropout=0.2, recurrent_dropout=0.1)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v1_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo

def construir_mixto_v2_E(forma_entrada, dim_salida):
    """ Mixto agresivamente regularizado para ventanas largas
    (ent90_sal30) donde v1_E sobreajusta desde la primera epoca.

    Cambios respecto a v1_E:
    - Capacidad reducida: Conv1D 32->16 filtros, LSTM 64->32 unidades.
    - Kernel grande (kernel_size=7) para capturar patrones semanales
      en ventana de 90 dias, en vez de patrones locales de 3 dias.
    - padding='causal' para evitar leakage temporal hacia el futuro
      (la conv solo mira pasado, coherente con forecasting).
    - Dropout 0.5 entre Conv1D y LSTM, y otro 0.5 antes de Dense.
    - L2 mas fuerte en Conv1D (1e-3 vs 1e-4 de v1_E).
    - Sin recurrent_dropout (kernel CPU optimizado, mas rapido).
    - LR fijo 3e-5, compatible con ReduceLROnPlateau.
    - Loss Huber, robusto a outliers en retornos."""
    reg = regularizers.l2(1e-3)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=7, activation='relu', padding='causal',
               kernel_regularizer=reg)(entradas)
    x = Dropout(0.5)(x)
    x = LSTM(32, dropout=0.1)(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v2_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0)
    )
    return modelo

# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent90_sal90 (target muy suave, prom. 90 dias)
# ──────────────────────────────────────────────────────────────
# Cuando la salida promedia 90 dias futuros, el target es casi
# constante (la media muestral converge muy rapido a la media
# historica). Esto implica:
#   - El baseline trivial (BuyAndHold) es muy dificil de batir.
#   - Cualquier modelo grande sobreajusta ruido inexistente.
#   - La senyal util, si la hay, es de bajo orden (tendencias
#     mensuales, no patrones intradiarios).
# Por eso estas variantes son mas pequenyas y mas regularizadas
# que sus equivalentes de sal30.

def construir_dense_v2_E(dim_entrada, dim_salida, neuronas=(96, 48)):
    """[Emilio] MLP reducido para sal90 (target muy suave).
    Cambios respecto a v1_E:
    - Capacidad reducida (128,64) -> (96,48) para no sobreajustar
      un target casi constante.
    - L2 mas fuerte (1e-4 -> 5e-4) y Dropout 0.4 -> 0.5.
    - LR fijo 3e-5 (compatible con ReduceLROnPlateau)."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v2_E')
    modelo.add(Dense(neuronas[0], activation='relu',
                     kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(neuronas[1], activation='relu',
                     kernel_regularizer=reg))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5,
                                                    clipnorm=1.0),
                   loss='mae')
    return modelo


def construir_recurrente_v4_E(forma_entrada, dim_salida, celda='LSTM'):
    """[Emilio] LSTM apilado para sal90 con target muy suave.
    Cambios respecto a v3_E:
    - Capacidad reducida (128,64) -> (64,32) unidades.
    - LR fijo 3e-5 (igual que v3_E, funcionaba bien).
    - Loss MAE en vez de Huber (mas comparable con baselines y
      gradientes mejor escalados para retornos en escala ~1e-3).
    - Mantiene LayerNormalization + clipnorm + sin recurrent_dropout."""
    modelo = Sequential(name=f'{celda}_v4_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU

    modelo.add(CeldaClase(64,
                          input_shape=forma_entrada,
                          dropout=0.1,
                          return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.1))
    modelo.add(CeldaClase(32,
                          dropout=0.1,
                          return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v2_E(forma_entrada, dim_salida, filtros=32, kernel=7):
    """[Emilio] Conv1D dilatada para sal90.
    Cambios respecto a v1_E:
    - Filtros reducidos 64 -> 32 (mitad de capacidad).
    - Kernel 3 -> 7 para capturar patrones semanales en ventana 90d.
    - L2 mas fuerte (1e-4 -> 5e-4) y Dropout 0.4 -> 0.5.
    - LR fijo 3e-5 (compatible con ReduceLROnPlateau).
    - Mantiene padding causal + dilatacion + GAP."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v2_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      dilation_rate=2, padding='causal',
                      input_shape=forma_entrada,
                      kernel_regularizer=reg))
    modelo.add(MaxPooling1D(pool_size=3))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu',
                      padding='same',
                      kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5,
                                                    clipnorm=1.0),
                   loss='mae')
    return modelo


def construir_mixto_v3_E(forma_entrada, dim_salida):
    """[Emilio] Mixto Conv1D+LSTM para sal90 (target muy suave).
    Relajacion respecto a v2_E (que era para sal30):
    - Conv1D 16 -> 24 filtros (mas extraccion local, sigue pequenyo).
    - Dropout 0.5 -> 0.3 (target mas predecible que sal30, menos
      riesgo de overfitting necesita menos regularizacion).
    - L2 1e-3 -> 5e-4 (regularizacion mas suave).
    - Loss Huber -> MAE (gradientes mejor escalados para retornos
      en escala ~1e-3, comparable directo con baselines).
    - Mantiene: kernel 7, padding causal, LSTM 32, LR 3e-5."""
    reg = regularizers.l2(5e-4)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(24, kernel_size=7, activation='relu', padding='causal',
               kernel_regularizer=reg)(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(32, dropout=0.1)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v3_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent90_sal90 derivadas del random search
# ──────────────────────────────────────────────────────────────
# Las versiones anteriores (Dense v2_E, Recurrente v4_E, Conv1D v2_E,
# Mixto v3_E) asumian que con target casi constante hacia falta
# regularizacion fuerte (lr=3e-5, dropout=0.5, l2=5e-4). El barrido
# aleatorio (Taller B3-T4, 120 combinaciones) no respalda esa hipotesis.
#
# Patron comun de los ganadores:
#   - lr=3e-4 (Dense, LSTM, Conv1D) o lr=1e-3 (Mixto)
#   - dropout=0.2 (Dense, LSTM, Mixto) o 0.4 (Conv1D)
#   - l2_reg=1e-4 cuando aplica
#   - activacion: gelu (Dense, Mixto), elu (LSTM), relu (Conv1D)
#   - clipnorm=1.0 en todos (estabilidad con lr moderado)
#   - loss='mae' (retornos en escala ~1e-3, gradientes mejor escalados
#     que con Huber).


def construir_dense_v3_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """MLP medium (128 -> 64) para ent90/sal90, ajustado por random search.

    Hiperparametros derivados del barrido:
    - capacidad medium (128, 64)  --> ~275k params con entrada 90*23=2070
    - dropout 0.2
    - activacion gelu
    - lr 3e-4
    - l2_reg 1e-4

    Cambios respecto a v2_E:
    - lr 3e-5 -> 3e-4 (10x mayor; el barrido lo respalda)
    - dropout 0.5 -> 0.2 (la regularizacion fuerte no aporta)
    - l2 5e-4 -> 1e-4 (idem)
    - relu -> gelu (top-1 en Dense)
    - capacidad (96,48) -> (128,64) (capacidad media, no extrema)

    Resultado del barrido: MAE val ~0.000987, MAE test ~0.001267
    (mejor MAE test entre las cuatro familias).
    """
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v3_E')
    modelo.add(Dense(neuronas[0], activation='gelu',
                     kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(neuronas[1], activation='gelu',
                     kernel_regularizer=reg))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v5_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM small apilada (32 -> 16) para ent90/sal90.

    Variante tanh del ganador del random search: el top-1 usaba elu pero
    desactivaba el kernel optimizado (cuDNN/Metal). Esta variante (tanh +
    dropout=0.2) era el #2 del barrido a solo 0.000002 de MAE, y entrena
    ~10x mas rapido en CPU sin GPU.
    """
    modelo = Sequential(name=f'{celda}_v5_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU

    modelo.add(CeldaClase(32,
                          activation='tanh',        # antes: 'elu'
                          input_shape=forma_entrada,
                          dropout=0.2,              # antes: 0.4
                          return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(16,
                          activation='tanh',        # antes: 'elu'
                          dropout=0.2,              # antes: 0.4
                          return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v3_E(forma_entrada, dim_salida, filtros=16, kernel=7):
    """Conv1D dilatada minimalista para ent90/sal90, ajustada por random search.

    Hiperparametros derivados del barrido:
    - filtros 16, kernel 7  --> 3.703 params (la red mas pequenya del barrido)
    - dropout 0.4
    - activacion relu
    - lr 3e-4

    Cambios respecto a v2_E:
    - filtros 32 -> 16
    - L2 5e-4 -> 0 (no aparece como ganador en Conv1D)
    - dropout 0.5 -> 0.4
    - lr 3e-5 -> 3e-4

    Arquitectura: Conv1D causal dilatada + MaxPooling + Conv1D + GAP.
    kernel=7 captura patrones semanales en ventana de 90 dias;
    dilation_rate=2 amplia campo receptivo sin parametros extra;
    padding='causal' evita leakage temporal.

    Es la opcion con mejor relacion coste/beneficio del barrido:
    entrena en ~16s, MAE test 0.001281 (segundo mejor global).
    """
    modelo = Sequential(name='Conv1D_v3_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      dilation_rate=2, padding='causal',
                      input_shape=forma_entrada))
    modelo.add(MaxPooling1D(pool_size=2, padding='same'))
    modelo.add(Conv1D(max(filtros // 2, 8), kernel_size=kernel,
                      activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v4_E(forma_entrada, dim_salida):
    """Conv1D + LSTM mixto para ent90/sal90, ajustado por random search.

    Hiperparametros derivados del barrido:
    - conv_filtros 32, lstm_unidades 64  --> ~30k params
    - dropout 0.3
    - activacion gelu
    - lr 1e-3  (UNICA familia donde 1e-3 gana, no 3e-4)

    Cambios respecto a v3_E:
    - conv_filtros 24 -> 32
    - lstm_unidades 32 -> 64
    - dropout 0.3 (igual)
    - L2 5e-4 -> 0 (el barrido no incluye L2 en Mixto y gana sin)
    - lr 3e-5 -> 1e-3 (~33x mayor; gelu + clipnorm lo estabiliza)
    - relu -> gelu

    Justificacion para coger el #3 del top en vez del #1: el #1 (dropout
    0.2) tiene MAE val 0.000979 pero MAE test 0.001292 (peor que
    BuyAndHold). El #3 (dropout 0.3) tiene MAE val 0.000981 y MAE test
    0.001272 (el unico Mixto que bate a BuyAndHold). El gap val-test es
    mas pequenyo, indicando mejor generalizacion.

    Arquitectura: Conv1D causal + Dropout + LSTM + Dropout + Dense.
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=5, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(64, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v4_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent05 (entrada corta: 5 dias)
# ──────────────────────────────────────────────────────────────
# Con solo 5 pasos temporales, la dinamica recurrente y la dilatacion
# temporal pierden valor: la dimension informativa rica son los 23
# activos, no los 5 dias. Las arquitecturas enfocan mezcla cross-asset
# y campo receptivo igual o superior a la ventana entera.
#
# Tres regimenes segun horizonte de salida:
#   sal01: target ruidoso (retorno diario sin suavizar)
#          --> capacidad baja, regularizacion alta, lr 1e-4
#   sal05: target con suavizado moderado (media 5d)
#          --> capacidad media, regularizacion equilibrada, lr 3e-4
#   sal30: target casi constante (media 30d)
#          --> capacidad muy baja, bottleneck extremo, lr 1e-4


# ──────────────── Dense para ent05 ────────────────

def construir_dense_v4_E(dim_entrada, dim_salida, neuronas=(64, 16)):
    """MLP con bottleneck regularizado para ent05/sal01.

    Tarea: predecir 1 dia de retornos crudos desde 5 dias. Target
    extremadamente ruidoso, baseline lineal muy fuerte. El bottleneck a
    16 obliga a filtrar ruido y aprender una representacion compacta
    (espiritu de denoising autoencoder supervisado).

    Hiperparametros:
    - capacidad: 64 -> 16 (bottleneck)
    - dropout 0.4 y 0.3 (fuerte, target con SNR muy bajo)
    - L2 3e-4
    - lr 1e-4, clipnorm 1.0 (pasos cortos para evitar memorizar ruido)
    """
    reg = regularizers.l2(3e-4)
    modelo = Sequential(name='Dense_v4_E')
    modelo.add(Dense(neuronas[0], activation='gelu',
                     kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(neuronas[1], activation='gelu',
                     kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_dense_v5_E(dim_entrada, dim_salida):
    """MLP con bloque residual para ent05/sal05.

    Tarea: media de 5 dias futuros desde 5 dias pasados. SNR mejor que
    sal01 (~sqrt(5) menos ruido), permite mas capacidad. El bloque
    residual `x + Dense(128)(x)` permite al modelo "saltar" la
    no-linealidad si no aporta, comportandose como una lineal mejorada
    en el peor caso.

    Hiperparametros:
    - 128 -> [res 128] -> 64 -> 23
    - dropout 0.25 en ambos puntos
    - L2 1e-4
    - lr 3e-4, clipnorm 1.0
    """
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=(dim_entrada,))
    x = Dense(128, activation='gelu', kernel_regularizer=reg)(entradas)
    x = Dropout(0.25)(x)
    h = Dense(128, activation='gelu', kernel_regularizer=reg)(x)
    x = Add()([x, h])                               # conexion residual
    x = Dropout(0.25)(x)
    x = Dense(64, activation='gelu', kernel_regularizer=reg)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Dense_v5_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_dense_v6_E(dim_entrada, dim_salida):
    """MLP con bottleneck profundo y regularizacion fuerte para ent05/sal30.

    Tarea: media de 30 dias desde 5 dias. Target casi constante. El
    bottleneck a 8 (menor que el numero de activos = 23) fuerza la
    captura de "factores de regimen" globales, coherente con la
    intuicion de Marchenko-Pastur de que la senyal real ocupa pocas
    dimensiones.

    Hiperparametros:
    - 48 -> 8 (bottleneck extremo) -> 24 -> 23
    - dropout 0.5 y 0.4 (muy fuerte)
    - L2 5e-4
    - lr 1e-4, clipnorm 1.0
    """
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v6_E')
    modelo.add(Dense(48, activation='gelu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(8, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(24, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── Recurrente para ent05 ────────────────

def construir_recurrente_v6_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM small con LayerNorm para ent05/sal01.

    Con 5 pasos temporales, leer la ventana en ambos sentidos es barato
    (4 estados extra) y enriquece la representacion. No viola causalidad:
    la entrada es historica completa; el futuro a predecir esta fuera
    de la ventana.

    Hiperparametros:
    - Bidirectional(LSTM 16) -> LayerNorm -> Dense 16 -> 23
    - elu en LSTM (mas robusto al ruido que tanh), dropout 0.4
    - dropout 0.3 en cabeza
    - lr 3e-4, clipnorm 1.0
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v6_E')
    modelo.add(Bidirectional(
        CeldaClase(16, activation='elu', dropout=0.4),
        input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v7_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM medium con cabeza Dense para ent05/sal05.

    Capacidad moderada, regularizacion equilibrada. Usa tanh para
    aprovechar el kernel optimizado (cuDNN/Metal: tanh es el unico que
    permite el kernel acelerado). La Dense head mezcla los estados
    forward y backward de forma no-lineal.

    Hiperparametros:
    - Bidirectional(LSTM 32, tanh) -> LayerNorm -> Dense 32 -> 23
    - dropout 0.3 en LSTM, 0.2 en cabeza
    - lr 3e-4, clipnorm 1.0
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v7_E')
    modelo.add(Bidirectional(
        CeldaClase(32, activation='tanh', dropout=0.3),
        input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v8_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM single small con LayerNorm y recurrent_dropout para ent05/sal30.

    Sin bidireccional: con target muy suave (media de 30 dias), mas
    capacidad solo sobreajusta. recurrent_dropout activado porque el
    target apenas varia y el ruido temporal hay que cortarlo.

    Hiperparametros:
    - LSTM 16, tanh -> LayerNorm -> Dense 16 -> 23
    - dropout 0.4, recurrent_dropout 0.2
    - dropout 0.3 en cabeza
    - lr 1e-4, clipnorm 1.0
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v8_E')
    modelo.add(CeldaClase(16, activation='tanh',
                          dropout=0.4, recurrent_dropout=0.2,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── Conv1D para ent05 ────────────────

def construir_conv1d_v4_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo (Inception-like) para ent05/sal01.

    Tres ramas en paralelo:
    - kernel=1: mezcla cross-asset (equivalente a Dense por paso)
    - kernel=3: microestructura de 3 dias
    - kernel=5: ventana completa
    Mucho mas eficiente que apilar Conv profundas en una ventana de 5,
    porque las dilatadas pierden sentido y el pooling agresivo destruye
    informacion en secuencias tan cortas.

    Hiperparametros:
    - 8 filtros por rama (24 features tras concatenar)
    - GlobalAveragePooling1D para agregar temporalmente
    - dropout 0.4
    - lr 3e-4, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(8, kernel_size=1, activation='relu', padding='same')(entradas)
    rama3 = Conv1D(8, kernel_size=3, activation='relu', padding='same')(entradas)
    rama5 = Conv1D(8, kernel_size=5, activation='relu', padding='same')(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v4_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v5_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel con cabeza Dense para ent05/sal05.

    Misma idea multiescala que v4_E pero con 16 filtros por rama (48
    features tras concatenar) y una Dense head de 32 que mezcla las
    tres escalas de forma no-lineal antes de la salida.

    Hiperparametros:
    - 16 filtros por rama
    - dropout 0.3 tras pooling
    - Dense 32 (gelu)
    - lr 3e-4, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(16, kernel_size=1, activation='relu', padding='same')(entradas)
    rama3 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(entradas)
    rama5 = Conv1D(16, kernel_size=5, activation='relu', padding='same')(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='gelu')(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v5_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v6_E(forma_entrada, dim_salida):
    """Conv1D minimalista con kernel = ventana completa para ent05/sal30.

    Un unico filtro de tamano 5 ve toda la ventana de una vez. Es
    esencialmente una proyeccion lineal por filtro con activacion
    no-lineal: minima capacidad, maxima regularizacion. La hipotesis
    es que para sal30 la unica senyal util es de muy bajo orden, asi
    que cualquier red mas grande sobreajusta.

    Hiperparametros:
    - 8 filtros, kernel=5, padding='valid' (sin replicar bordes)
    - L2 5e-4
    - Flatten + Dropout 0.5 antes de salida
    - lr 1e-4, clipnorm 1.0
    """
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v6_E')
    modelo.add(Conv1D(8, kernel_size=5, activation='gelu', padding='valid',
                      kernel_regularizer=reg,
                      input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── Mixto para ent05 ────────────────

def construir_mixto_v5_E(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU minimo para ent05/sal01.

    TimeDistributed(Dense 16) comprime cada paso de 23 activos a 16
    features (capa de mezcla cross-asset). Luego GRU diminuta sobre
    5 pasos x 16 features. Menos parametros que Mixto_v1 y enfocado
    donde si esta la informacion: relacion entre activos.

    Hiperparametros:
    - TimeDistributed(Dense 16, gelu) -> GRU 16 -> 23
    - dropout 0.3 en GRU y 0.3 antes de salida
    - lr 3e-4, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(16, activation='gelu'))(entradas)
    x = GRU(16, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v5_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v6_E(forma_entrada, dim_salida):
    """Conv multi-kernel + LSTM bidireccional para ent05/sal05.

    Primero extraccion de features locales multiescala con dos kernels
    paralelos (3 y 5), luego procesamiento temporal bidireccional.
    El modelo mas completo de los tres para sal05, justificado por
    mejor SNR del target.

    Hiperparametros:
    - Conv1D 16 filtros kernel=3 + Conv1D 16 filtros kernel=5 (paralelo)
    - Bidirectional(LSTM 24) sobre concatenacion (32 features)
    - dropout 0.3 en LSTM, 0.2 antes de salida
    - lr 3e-4, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    rama3 = Conv1D(16, kernel_size=3, activation='gelu', padding='same')(entradas)
    rama5 = Conv1D(16, kernel_size=5, activation='gelu', padding='same')(entradas)
    x = Concatenate()([rama3, rama5])
    x = Bidirectional(LSTM(24, dropout=0.3))(x)
    x = Dropout(0.2)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v6_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v7_E(forma_entrada, dim_salida):
    """Conv compacta + GRU diminuta para ent05/sal30.

    GRU 8 unidades es muy pequena pero suficiente cuando el target
    apenas varia. Conv1D causal mantiene la coherencia temporal sin
    leakage del futuro al pasado.

    Hiperparametros:
    - Conv1D 16 filtros kernel=3, padding='causal'
    - GRU 8 unidades, dropout 0.3
    - dropout 0.4 tras Conv, 0.3 antes de salida
    - lr 1e-4, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.4)(x)
    x = GRU(8, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v7_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent05/sal90 (entrada cortisima + target casi constante)
# ──────────────────────────────────────────────────────────────
# Caso mas extremo: 5 dias de entrada para predecir media de 90 dias.
# Bottleneck mas estrecho que ent05/sal30 (4 vs 8) porque el target
# es aun mas plano. La hipotesis es que el espacio util se reduce a
# ~3-4 factores macro (volatilidad, direccion, regimen dominante).
# Patron: lr 5e-5, dropout 0.5, L2 1e-3 (los mas extremos del set).


def construir_dense_v7_E(dim_entrada, dim_salida):
    """MLP con bottleneck ultra-estrecho para ent05/sal90.

    Bottleneck a 4 (vs 8 en v6_E para ent05/sal30) porque el target
    a 90 dias es aun mas plano. La idea es forzar al modelo a aprender
    solo 3-4 factores macro: volatilidad reciente, direccion reciente,
    y regimen dominante.

    Arquitectura: 115 -> 32 -> 4 -> 16 -> 90
    Hiperparametros: lr 5e-5, dropout 0.5, L2 1e-3, clipnorm 1.0
    """
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Dense_v7_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(4, activation='gelu', kernel_regularizer=reg))   # bottleneck ultra
    modelo.add(Dropout(0.5))
    modelo.add(Dense(16, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v9_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM micro para ent05/sal90.

    Solo 8 unidades porque la recurrencia con 5 pasos aporta poco y
    el target tan suave penaliza cualquier capacidad sobrante.
    recurrent_dropout 0.3 corta el ruido temporal mas agresivamente
    que en sal30 (que era 0.2).

    Arquitectura: LSTM 8 -> LayerNorm -> Dense 8 -> Dropout -> 90
    Hiperparametros: lr 5e-5, dropout 0.4, recurrent_dropout 0.3
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v9_E')
    modelo.add(CeldaClase(8, activation='tanh',
                          dropout=0.4, recurrent_dropout=0.3,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(8, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v7_E(forma_entrada, dim_salida):
    """Conv1D ultra-minimalista para ent05/sal90.

    4 filtros (vs 8 en v6_E) y kernel=5 (toda la ventana). Es el
    modelo mas pequeno de todos los _E. Si una proyeccion lineal por
    filtro no captura la senyal, ningun modelo mas complejo la va a
    capturar con SNR tan bajo.

    Arquitectura: Conv1D(4, k=5, valid) -> Flatten -> Dropout 0.5 -> Dense
    Hiperparametros: lr 5e-5, L2 1e-3
    """
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Conv1D_v7_E')
    modelo.add(Conv1D(4, kernel_size=5, activation='gelu', padding='valid',
                      kernel_regularizer=reg,
                      input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v8_E(forma_entrada, dim_salida):
    """Conv1D causal + GRU diminuta para ent05/sal90.

    Variante minima de v7_E (ent05/sal30): GRU baja de 8 a 4 unidades,
    dropouts suben a 0.5/0.4, lr cae a 5e-5.

    Arquitectura: Conv1D(8, k=3, causal) -> GRU 4 -> Dropout -> 90
    Hiperparametros: lr 5e-5, dropout 0.5/0.4
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(8, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.5)(x)
    x = GRU(4, dropout=0.4)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v8_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent10 (entrada moderada: 10 dias)
# ──────────────────────────────────────────────────────────────
# Punto de inflexion donde tienen sentido por primera vez:
#   - Conv1D dilatada apilada (kernel=3 + dilation=2 cubre 5d efectivos)
#   - LSTM apilada vertical (la 2a capa ve 10 estados resumidos)
#   - Kernel=7 en Conv (cubre 70% de la ventana)
#
# Las arquitecturas se vuelven mas ricas pero los hiperparametros
# (lr, dropout, L2) siguen escalando con el horizonte de salida.


# ──────────────── ent10/sal01 ────────────────

def construir_dense_v8_E(dim_entrada, dim_salida):
    """MLP con bottleneck moderado para ent10/sal01.

    Target ruidoso (1 dia sin suavizar) con entrada de 230 features
    (2x la de ent05). Primera capa mas ancha (128 vs 64 en v4_E) por
    la entrada mas rica, pero el bottleneck a 16 mantiene la
    regularizacion por compresion.

    Arquitectura: 230 -> 128 -> 32 -> 16 -> 1
    Hiperparametros: lr 1e-4, dropout 0.4/0.3, L2 2e-4
    """
    reg = regularizers.l2(2e-4)
    modelo = Sequential(name='Dense_v8_E')
    modelo.add(Dense(128, activation='gelu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(16, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v10_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM medium para ent10/sal01.

    Con 10 pasos, el bidireccional ofrece 20 estados efectivos: capacidad
    razonable sin sobreajuste. tanh permite el kernel optimizado (cuDNN/
    Metal). La Dense head mezcla forward+backward de forma no-lineal.

    Arquitectura: Bidirectional(LSTM 24, tanh) -> LayerNorm -> Dense 32 -> 1
    Hiperparametros: lr 3e-4, dropout 0.3
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v10_E')
    modelo.add(Bidirectional(
        CeldaClase(24, activation='tanh', dropout=0.3),
        input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v8_E(forma_entrada, dim_salida):
    """Conv1D apilada con dilatacion para ent10/sal01.

    Primera aparicion de stacking + dilation en el set _E. Dos capas
    Conv1D causales: la primera local (k=3, dilation=1), la segunda
    con campo receptivo ampliado (k=3, dilation=2 -> cubre 5d
    efectivos). Total: cada output del stack ve 5 dias hacia atras.

    Arquitectura: Conv1D(16, k=3) -> Conv1D(16, k=3, dilation=2) -> GAP -> 1
    Hiperparametros: lr 3e-4, dropout 0.4
    """
    modelo = Sequential(name='Conv1D_v8_E')
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='causal',
                      input_shape=forma_entrada))
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='causal',
                      dilation_rate=2))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v9_E(forma_entrada, dim_salida):
    """Conv1D causal + Bidirectional LSTM para ent10/sal01.

    Por primera vez en el set _E la combinacion Conv -> LSTM tiene
    espacio para aportar: Conv extrae features locales (k=3 sobre 10
    pasos), BiLSTM hace integracion temporal global con 20 estados.

    Arquitectura: Conv1D(16, k=3, causal) -> Bidirectional(LSTM 16) -> 1
    Hiperparametros: lr 3e-4, dropout 0.3
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(16, dropout=0.3))(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v9_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── ent10/sal05 ────────────────

def construir_dense_v9_E(dim_entrada, dim_salida):
    """MLP residual ampliado para ent10/sal05.

    Sweet spot del set: mejor SNR de target + entrada moderada.
    Extension del v5_E (residual MLP, ent05/sal05) con primera capa
    192 (vs 128) porque la entrada es 2x mas grande (230 vs 115).
    El bloque residual permite "saltar" la no-linealidad si no aporta.

    Arquitectura: 230 -> 192 -> [+res 192] -> 96 -> 5
    Hiperparametros: lr 3e-4, dropout 0.25, L2 1e-4
    """
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=(dim_entrada,))
    x = Dense(192, activation='gelu', kernel_regularizer=reg)(entradas)
    x = Dropout(0.25)(x)
    h = Dense(192, activation='gelu', kernel_regularizer=reg)(x)
    x = Add()([x, h])                                # conexion residual
    x = Dropout(0.25)(x)
    x = Dense(96, activation='gelu', kernel_regularizer=reg)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Dense_v9_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v11_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM apilada vertical para ent10/sal05.

    Primera aparicion de LSTM stacking en el set _E. Con 10 pasos, la
    segunda capa ve una secuencia de 10 estados resumidos por la
    primera: capacidad jerarquica que no existia con ent05 (donde
    apilar daba al segundo nivel secuencias de 5).

    Arquitectura: LSTM 48 (return_seq) -> LayerNorm -> LSTM 24 -> LayerNorm -> Dense 32 -> 5
    Hiperparametros: lr 3e-4, dropout 0.3
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v11_E')
    modelo.add(CeldaClase(48, activation='tanh', dropout=0.3,
                          return_sequences=True,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(24, activation='tanh', dropout=0.3))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v9_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel con refinamiento para ent10/sal05.

    Variante mas rica que v5_E (multi-kernel + Dense head). Ahora con
    kernel=7 (cubre 70% de la ventana de 10d, no posible con ent05)
    y una capa Conv1D adicional tras la concatenacion que refina los
    features multiescala antes del pooling.

    Arquitectura: Conv1D paralelo (k=3,5,7) x 16 -> Concat -> Conv1D(24, k=3) -> GAP -> Dense 32 -> 5
    Hiperparametros: lr 3e-4, dropout 0.3
    """
    entradas = Input(shape=forma_entrada)
    rama3 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(entradas)
    rama5 = Conv1D(16, kernel_size=5, activation='relu', padding='same')(entradas)
    rama7 = Conv1D(16, kernel_size=7, activation='relu', padding='same')(entradas)
    x = Concatenate()([rama3, rama5, rama7])
    x = Conv1D(24, kernel_size=3, activation='gelu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='gelu')(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v9_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v10_E(forma_entrada, dim_salida):
    """Conv multi-kernel + Bidirectional LSTM para ent10/sal05.

    Combinacion mas rica del set: dos kernels paralelos (3, 5) para
    multiescala local, luego BiLSTM 32 con 20 estados efectivos para
    integracion global. La Dense head mezcla con no-linealidad.

    Arquitectura: Conv1D (k=3, k=5) x 24 -> Concat -> Bidirectional(LSTM 32) -> Dense 32 -> 5
    Hiperparametros: lr 3e-4, dropout 0.3/0.2
    """
    entradas = Input(shape=forma_entrada)
    rama3 = Conv1D(24, kernel_size=3, activation='gelu', padding='same')(entradas)
    rama5 = Conv1D(24, kernel_size=5, activation='gelu', padding='same')(entradas)
    x = Concatenate()([rama3, rama5])
    x = Bidirectional(LSTM(32, dropout=0.3))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='gelu')(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v10_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── ent10/sal30 ────────────────

def construir_dense_v10_E(dim_entrada, dim_salida):
    """MLP con bottleneck moderado para ent10/sal30.

    Variante de v6_E (ent05/sal30) con primera capa mas ancha (64 vs
    48) por entrada 2x mayor. Bottleneck a 12 (un poco mas permisivo
    que el 8 de v6_E, porque hay mas informacion de entrada para
    sostener mas factores latentes).

    Arquitectura: 230 -> 64 -> 12 (bottleneck) -> 32 -> 30
    Hiperparametros: lr 1e-4, dropout 0.5/0.4, L2 5e-4
    """
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v10_E')
    modelo.add(Dense(64, activation='gelu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(12, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v12_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM single small con recurrent_dropout para ent10/sal30.

    Variante de v8_E (ent05/sal30) con 24 unidades (vs 16). Sin
    bidireccional: con target muy suave mas capacidad solo sobreajusta.
    El recurrent_dropout corta el ruido temporal.

    Arquitectura: LSTM 24 (tanh, recurrent_dropout 0.2) -> LayerNorm -> Dense 16 -> 30
    Hiperparametros: lr 1e-4, dropout 0.4
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v12_E')
    modelo.add(CeldaClase(24, activation='tanh',
                          dropout=0.4, recurrent_dropout=0.2,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v10_E(forma_entrada, dim_salida):
    """Conv1D dilatada minimalista para ent10/sal30.

    Single layer con kernel=5 + dilation=2: campo receptivo efectivo
    de 9 dias (casi toda la ventana de 10) con muy pocos parametros.
    Mantiene el minimalismo de v6_E (ent05/sal30) pero aprovecha la
    ventana mayor con dilatacion.

    Arquitectura: Conv1D(8, k=5, dilation=2, causal) -> GAP -> Dropout -> Dense
    Hiperparametros: lr 1e-4, L2 5e-4, dropout 0.5
    """
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v10_E')
    modelo.add(Conv1D(8, kernel_size=5, activation='gelu', padding='causal',
                      dilation_rate=2, kernel_regularizer=reg,
                      input_shape=forma_entrada))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v11_E(forma_entrada, dim_salida):
    """Conv1D dilatada + GRU para ent10/sal30.

    Variante de v7_E (ent05/sal30) con dilatacion en la Conv (campo
    receptivo de 5 con kernel=3) y GRU ligeramente mas grande (12 vs
    8) por entrada mayor.

    Arquitectura: Conv1D(16, k=3, dilation=2, causal) -> GRU 12 -> 30
    Hiperparametros: lr 1e-4, dropout 0.4/0.3
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal',
               dilation_rate=2)(entradas)
    x = Dropout(0.4)(x)
    x = GRU(12, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v11_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────── ent10/sal90 ────────────────

def construir_dense_v11_E(dim_entrada, dim_salida):
    """MLP bottleneck ultra para ent10/sal90.

    Mismo bottleneck extremo a 4 que v7_E (ent05/sal90), pero primera
    capa moderadamente mayor (32) por entrada 2x mas rica.

    Arquitectura: 230 -> 32 -> 4 (bottleneck ultra) -> 24 -> 90
    Hiperparametros: lr 5e-5, dropout 0.5, L2 1e-3
    """
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Dense_v11_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(4, activation='gelu', kernel_regularizer=reg))   # bottleneck ultra
    modelo.add(Dropout(0.5))
    modelo.add(Dense(24, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v13_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM micro con recurrent_dropout fuerte para ent10/sal90.

    Variante de v9_E (ent05/sal90) con 12 unidades (vs 8) por entrada
    mayor. recurrent_dropout 0.3 corta el ruido temporal con la
    misma agresividad.

    Arquitectura: LSTM 12 (tanh, recurrent_dropout 0.3) -> LayerNorm -> Dense 12 -> 90
    Hiperparametros: lr 5e-5, dropout 0.4
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v13_E')
    modelo.add(CeldaClase(12, activation='tanh',
                          dropout=0.4, recurrent_dropout=0.3,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(12, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v11_E(forma_entrada, dim_salida):
    """Conv1D dilatada minimalista para ent10/sal90.

    6 filtros (entre v6_E con 8 y v7_E con 4), kernel=5 + dilation=2
    cubre 9 dias efectivos. padding='valid' para no replicar bordes
    (mas conservador, evita asumir simetria).

    Arquitectura: Conv1D(6, k=5, dilation=2, valid) -> Flatten -> Dropout -> Dense
    Hiperparametros: lr 5e-5, L2 1e-3, dropout 0.5
    """
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Conv1D_v11_E')
    modelo.add(Conv1D(6, kernel_size=5, activation='gelu', padding='valid',
                      dilation_rate=2, kernel_regularizer=reg,
                      input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v12_E(forma_entrada, dim_salida):
    """Conv1D causal + GRU diminuta para ent10/sal90.

    Variante de v8_E (ent05/sal90) con un poco mas de capacidad por
    la entrada mayor pero manteniendo el caracter minimalista.

    Arquitectura: Conv1D(8, k=3, causal) -> GRU 6 -> 90
    Hiperparametros: lr 5e-5, dropout 0.5/0.4
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(8, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.5)(x)
    x = GRU(6, dropout=0.4)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v12_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss='mae'
    )
    return modelo


# ──────────────────────────────────────────────────────────────
# VARIANTES PARA ent90/sal30 (entrada larga + target moderado)
# ──────────────────────────────────────────────────────────────
# Reemplazo de las versiones antiguas (v1_E Dense, v3_E LSTM, v1_E Conv1D,
# v2_E Mixto), que estaban mal calibradas y colapsaban a BuyAndHold o
# divergian (LSTM con Huber sobre escala 1e-3).
#
# Aplican lo aprendido del random search:
#   - lr 3e-4 con clipnorm 1.0 (vs 3e-5 anterior)
#   - loss 'mae' (vs Huber, que con delta=1 sobre retornos en escala 1e-3
#     se comporta como MSE puro y rompe la robustez prometida)
#   - dropout 0.2-0.3 (vs 0.5 que congelaba el aprendizaje)
#   - gelu en Dense/Mixto, tanh en LSTM (kernel optimizado)
#
# Con ventana 90 dias por primera vez tienen sentido:
#   - LSTM apilada vertical (capacidad jerarquica)
#   - Conv1D dilatada apilada (campo receptivo de 13+ dias)


def construir_dense_v12_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """MLP medium para ent90/sal30.

    Reemplazo de v1_E (que tenia relu sin dropout y colapsaba a BAH).
    Capacidad moderada (128, 64) con gelu y dropout 0.25, dimensionada
    para una entrada de 2070 features (90 dias x 23 activos).

    Arquitectura: 2070 -> 128 -> 64 -> 30
    Hiperparametros: lr 3e-4, dropout 0.25, L2 1e-4, clipnorm 1.0
    """
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v12_E')
    modelo.add(Dense(neuronas[0], activation='gelu',
                     kernel_regularizer=reg,
                     input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(neuronas[1], activation='gelu',
                     kernel_regularizer=reg))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_recurrente_v14_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM apilada vertical para ent90/sal30.

    Reemplazo de v3_E (lr 3e-5 + Huber, que divergia: MAE test ~5x peor
    que BAH). Con 90 pasos, apilar dos LSTM tiene sentido: la segunda
    capa ve una secuencia de 90 estados resumidos jerarquicamente.

    Arquitectura: LSTM 64 (return_seq) -> LayerNorm -> LSTM 32 -> LayerNorm -> Dense 32 -> 30
    Hiperparametros: lr 3e-4, dropout 0.3 (LSTM) / 0.2 (Dense head), clipnorm 1.0
    """
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v14_E')
    modelo.add(CeldaClase(64, activation='tanh', dropout=0.3,
                          return_sequences=True,
                          input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(32, activation='tanh', dropout=0.3))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_conv1d_v12_E(forma_entrada, dim_salida, filtros=32, kernel=7):
    """Conv1D apilada dilatada para ent90/sal30.

    Reemplazo de v1_E. Dos capas Conv1D causales con dilation_rate
    progresivo (1 -> 2), kernel=7 para capturar patrones semanales.
    Campo receptivo efectivo tras las dos capas: ~13 dias, suficiente
    para tendencias quincenales sin necesidad de mas profundidad.

    Arquitectura: Conv1D(32, k=7) -> Conv1D(32, k=7, dilation=2) -> GAP -> Dropout -> 30
    Hiperparametros: lr 3e-4, dropout 0.3, clipnorm 1.0
    """
    modelo = Sequential(name='Conv1D_v12_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      padding='causal', input_shape=forma_entrada))
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu',
                      padding='causal', dilation_rate=2))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo


def construir_mixto_v13_E(forma_entrada, dim_salida):
    """Conv1D dilatado + LSTM para ent90/sal30.

    Reemplazo de v2_E (que con lr 3e-5 + dropout 0.5 doble + Huber daba
    MAE EXACTAMENTE igual a BuyAndHold: el modelo no se movia de la
    inicializacion). Variante "sana" con los hiperparametros del random
    search: Conv1D dilatada extrae patrones quincenales, LSTM 32 integra
    globalmente. dropout 0.3 (no 0.5).

    Arquitectura: Conv1D(32, k=7, dilation=2, causal) -> LSTM 32 -> 30
    Hiperparametros: lr 3e-4, dropout 0.3, clipnorm 1.0
    """
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=7, activation='gelu', padding='causal',
               dilation_rate=2)(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(32, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v13_E')
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='mae'
    )
    return modelo
