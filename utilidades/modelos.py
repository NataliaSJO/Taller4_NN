import keras
from keras.models import Sequential, Model
from keras.layers import (
    Dense, LSTM, GRU, Conv1D, GlobalAveragePooling1D,
    Flatten, Input, Dropout, MaxPooling1D, Bidirectional,
    SeparableConv1D
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
