import keras
from keras.models import Sequential, Model
from keras.layers import (
    Dense, LSTM, GRU, Conv1D, GlobalAveragePooling1D,
    Flatten, Input, Dropout, MaxPooling1D, Bidirectional, LayerNormalization,
    SeparableConv1D, Concatenate, TimeDistributed, Add
)
from keras import regularizers

# ==============================================================================
# 1. MODELOS BASELINE
# ==============================================================================

def construir_baseline_lineal(dim_entrada, dim_salida):
    """Capa Dense sin activación — equivalente a regresión lineal en Keras."""
    modelo = Sequential(name='Lineal_Keras')
    modelo.add(Dense(dim_salida, input_shape=(dim_entrada,)))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


# ==============================================================================
# 2. FAMILIA DENSA (MLP)
# Solo capas Dense + Dropout. Sin BatchNorm ni capas recurrentes/conv.
# ==============================================================================

# --- Versiones Base ---

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

def construir_dense_v7(dim_entrada, dim_salida):
    """MLP refinado para ent30/sal30 (heredero de Dense_v4). 128->LayerNorm->64. gelu, L2 1e-4, lr 1e-4."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v7')
    modelo.add(Dense(128, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v8(dim_entrada, dim_salida):
    """MLP refinado para ent30/sal90 (capacidad mayor que v7 para target 90d). 192->LayerNorm->96. gelu, L2 2e-4."""
    reg = regularizers.l2(2e-4)
    modelo = Sequential(name='Dense_v8')
    modelo.add(Dense(192, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(96, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v9(dim_entrada, dim_salida):
    """MLP anti-overfit reforzado para ent30/sal01 (SNR cero + 30 dias input). Bottleneck 64->16, gelu, L2 5e-4, lr 5e-5."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v9')
    modelo.add(Dense(64, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(16, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

# --- Variantes de Emilio (_E) ---

def construir_dense_v1_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """[Emilio] MLP reducido (128->64) con L2 + Dropout 0.4 + LR 3e-4."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v1_E')
    modelo.add(Dense(neuronas[0], activation='relu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(neuronas[1], activation='relu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo

def construir_dense_v2_E(dim_entrada, dim_salida, neuronas=(96, 48)):
    """[Emilio] MLP reducido para sal90 (target muy suave)."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v2_E')
    modelo.add(Dense(neuronas[0], activation='relu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(neuronas[1], activation='relu', kernel_regularizer=reg))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v3_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """MLP medium (128 -> 64) para ent90/sal90, ajustado por random search."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v3_E')
    modelo.add(Dense(neuronas[0], activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(neuronas[1], activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v4_E(dim_entrada, dim_salida, neuronas=(64, 16)):
    """MLP con bottleneck regularizado para ent05/sal01."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v4_E')
    modelo.add(Dense(neuronas[0], activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(neuronas[1], activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v5_E(dim_entrada, dim_salida):
    """MLP con bloque residual para ent05/sal05."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=(dim_entrada,))
    x = Dense(128, activation='gelu', kernel_regularizer=reg)(entradas)
    x = Dropout(0.25)(x)
    h = Dense(128, activation='gelu', kernel_regularizer=reg)(x)
    x = Add()([x, h])                               
    x = Dropout(0.25)(x)
    x = Dense(64, activation='gelu', kernel_regularizer=reg)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Dense_v5_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v6_E(dim_entrada, dim_salida):
    """MLP con bottleneck profundo y regularizacion fuerte para ent05/sal30."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v6_E')
    modelo.add(Dense(48, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(8, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(24, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v7_E(dim_entrada, dim_salida):
    """MLP con bottleneck ultra-estrecho para ent05/sal90."""
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Dense_v7_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(4, activation='gelu', kernel_regularizer=reg))   
    modelo.add(Dropout(0.5))
    modelo.add(Dense(16, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v8_E(dim_entrada, dim_salida):
    """MLP con bottleneck moderado para ent10/sal01."""
    reg = regularizers.l2(2e-4)
    modelo = Sequential(name='Dense_v8_E')
    modelo.add(Dense(128, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(16, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v9_E(dim_entrada, dim_salida):
    """MLP residual ampliado para ent10/sal05."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=(dim_entrada,))
    x = Dense(192, activation='gelu', kernel_regularizer=reg)(entradas)
    x = Dropout(0.25)(x)
    h = Dense(192, activation='gelu', kernel_regularizer=reg)(x)
    x = Add()([x, h])                                
    x = Dropout(0.25)(x)
    x = Dense(96, activation='gelu', kernel_regularizer=reg)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Dense_v9_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v10_E(dim_entrada, dim_salida):
    """MLP con bottleneck moderado para ent10/sal30."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v10_E')
    modelo.add(Dense(64, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(12, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v11_E(dim_entrada, dim_salida):
    """MLP bottleneck ultra para ent10/sal90."""
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Dense_v11_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(4, activation='gelu', kernel_regularizer=reg))   
    modelo.add(Dropout(0.5))
    modelo.add(Dense(24, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v12_E(dim_entrada, dim_salida, neuronas=(128, 64)):
    """MLP medium para ent90/sal30."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Dense_v12_E')
    modelo.add(Dense(neuronas[0], activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(neuronas[1], activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v13_E(dim_entrada, dim_salida):
    """MLP con bottleneck ULTRA-estrecho para ent05/sal01."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v13_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(4, activation='gelu', kernel_regularizer=reg))    
    modelo.add(Dropout(0.5))
    modelo.add(Dense(12, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v13_E_bis(dim_entrada, dim_salida):
    """MLP bottleneck ultra-regularizado para ent05/sal01 (anti-overfit)."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v13_E_bis')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(8, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v16_E(dim_entrada, dim_salida, neuronas=(48, 12)):
    """MLP bottleneck para ent05/sal05."""
    reg = regularizers.l2(3e-4)
    modelo = Sequential(name='Dense_v16_E')
    modelo.add(Dense(neuronas[0], activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(neuronas[1], activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v17_E(dim_entrada, dim_salida):
    """MLP con skip-residual desde el input para ent05/sal30."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=(dim_entrada,))
    x = Dense(96, activation='gelu', kernel_regularizer=reg)(entradas)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='gelu', kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)
    rama_nolineal = Dense(dim_salida)(x)
    rama_lineal = Dense(dim_salida, use_bias=False, name='proyeccion_lineal')(entradas)
    salidas = Add()([rama_lineal, rama_nolineal])
    modelo = Model(inputs=entradas, outputs=salidas, name='Dense_v17_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v18_E(dim_entrada, dim_salida):
    """MLP wide con LayerNorm para ent05/sal90."""
    reg = regularizers.l2(2e-4)
    modelo = Sequential(name='Dense_v18_E')
    modelo.add(Dense(128, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v19_E(dim_entrada, dim_salida):
    """MLP bottleneck ultra-regularizado para ent10/sal01 (anti-overfit, hereda de v13_E)."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Dense_v19_E')
    modelo.add(Dense(32, activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(8, activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_dense_v20_E(dim_entrada, dim_salida, neuronas=(64, 16)):
    """MLP bottleneck para ent10/sal05 (capacidad media, hereda de v16_E)."""
    reg = regularizers.l2(3e-4)
    modelo = Sequential(name='Dense_v20_E')
    modelo.add(Dense(neuronas[0], activation='gelu', kernel_regularizer=reg, input_shape=(dim_entrada,)))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(neuronas[1], activation='gelu', kernel_regularizer=reg))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

# ==============================================================================
# 3. FAMILIA RECURRENTE (LSTM / GRU)
# Solo celdas LSTM / GRU (incluido Bidirectional). Sin Conv1D.
# ==============================================================================

# --- Versiones Base ---

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
    modelo.add(Bidirectional(GRU(32, return_sequences=False), input_shape=forma_entrada))
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
    modelo.add(LSTM(128, input_shape=forma_entrada, dropout=0.2, recurrent_dropout=0.1, return_sequences=False))
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

def construir_recurrente_v7(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM + Dense intermedia para ent30/sal30 (heredero de LSTM_v4). 64+LayerNorm+Dense32. cuDNN/Metal activo."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v7')
    modelo.add(CeldaClase(64, activation='tanh', dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v8(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM + Dense intermedia para ent30/sal90. Dense head 48 para proyectar a 90d. cuDNN/Metal activo."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v8')
    modelo.add(CeldaClase(64, activation='tanh', dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(48, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v9(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM anti-overfit para ent30/sal01 (30 pasos + SNR cero). LSTM 16+LayerNorm+Dense 12. Mantiene recurrent_dropout=0.3 (sin cuDNN)."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v9')
    modelo.add(CeldaClase(16, activation='tanh', dropout=0.2, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(12, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

# --- Variantes de Emilio (_E) ---

def construir_recurrente_v1_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """[Emilio] LSTM/GRU monocapa con dropout interno (0.2 + 0.2) y LR 1e-4."""
    modelo = Sequential(name=f'{celda}_v1_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(unidades, input_shape=forma_entrada, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
    return modelo

def construir_recurrente_v2_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """[Emilio] LSTM apilada 2 capas (128 -> 64) para ventana de entrada 90d."""
    modelo = Sequential(name=f'{celda}_v2_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(128, input_shape=forma_entrada, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3, clipnorm=1.0), loss=keras.losses.Huber(delta=1.0))
    return modelo

def construir_recurrente_v3_E(forma_entrada, dim_salida, celda='LSTM', unidades=64):
    """[Emilio] LSTM apilada 2 capas (128 -> 64) sin recurrent_dropout."""
    modelo = Sequential(name=f'{celda}_v3_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(128, input_shape=forma_entrada, dropout=0.1, return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.1))
    modelo.add(CeldaClase(64, dropout=0.1, return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss=keras.losses.Huber(delta=1.0))
    return modelo

def construir_recurrente_v4_E(forma_entrada, dim_salida, celda='LSTM'):
    """[Emilio] LSTM apilado para sal90 con target muy suave."""
    modelo = Sequential(name=f'{celda}_v4_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(64, input_shape=forma_entrada, dropout=0.1, return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.1))
    modelo.add(CeldaClase(32, dropout=0.1, return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v5_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM small apilada (32 -> 16) para ent90/sal90, ajustada por random search."""
    modelo = Sequential(name=f'{celda}_v5_E')
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo.add(CeldaClase(32, activation='elu', input_shape=forma_entrada, dropout=0.4, return_sequences=True))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(16, activation='elu', dropout=0.4, return_sequences=False))
    modelo.add(LayerNormalization())
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v6_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM small con LayerNorm para ent05/sal01."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v6_E')
    modelo.add(Bidirectional(CeldaClase(16, activation='tanh', dropout=0.5), input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v7_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM medium con cabeza Dense para ent05/sal05."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v7_E')
    modelo.add(Bidirectional(CeldaClase(32, activation='tanh', dropout=0.3), input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v8_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM single small con LayerNorm y recurrent_dropout para ent05/sal30."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v8_E')
    modelo.add(CeldaClase(16, activation='tanh', dropout=0.4, recurrent_dropout=0.2, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v9_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM micro para ent05/sal90."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v9_E')
    modelo.add(CeldaClase(8, activation='tanh', dropout=0.4, recurrent_dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(8, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v10_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM medium para ent10/sal01."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v10_E')
    modelo.add(Bidirectional(CeldaClase(24, activation='tanh', dropout=0.3), input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v11_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM apilada vertical para ent10/sal05."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v11_E')
    modelo.add(CeldaClase(48, activation='tanh', dropout=0.3, return_sequences=True, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(24, activation='tanh', dropout=0.3))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v12_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM single small con recurrent_dropout para ent10/sal30."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v12_E')
    modelo.add(CeldaClase(24, activation='tanh', dropout=0.4, recurrent_dropout=0.2, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v13_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM micro con recurrent_dropout fuerte para ent10/sal90."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v13_E')
    modelo.add(CeldaClase(12, activation='tanh', dropout=0.4, recurrent_dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(12, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v14_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM apilada vertical para ent90/sal30."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v14_E')
    modelo.add(CeldaClase(64, activation='tanh', dropout=0.3, return_sequences=True, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(CeldaClase(32, activation='tanh', dropout=0.3))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v15_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM single ULTRA-pequenya para ent05/sal01."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v15_E')
    modelo.add(CeldaClase(4, activation='tanh', dropout=0.5, recurrent_dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(8, activation='gelu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v15_E_bis(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM unidireccional micro para ent05/sal01 (anti-overfit)."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v15_E_bis')
    modelo.add(CeldaClase(8, activation='tanh', dropout=0.5, recurrent_dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(8, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v16_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM unidireccional + LayerNorm para ent05/sal05."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v16_E')
    modelo.add(CeldaClase(16, activation='tanh', dropout=0.5, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(12, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v17_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM mediano para ent05/sal30."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v17_E')
    modelo.add(Bidirectional(CeldaClase(24, activation='tanh', dropout=0.3), input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(24, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v18_E(forma_entrada, dim_salida, celda='LSTM'):
    """Bidirectional LSTM con LayerNorm para ent05/sal90."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v18_E')
    modelo.add(Bidirectional(CeldaClase(20, activation='tanh', dropout=0.3), input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v19_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM unidireccional micro para ent10/sal01 (anti-overfit, hereda de v15_E, mantiene recurrent_dropout)."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v19_E')
    modelo.add(CeldaClase(10, activation='tanh', dropout=0.5, recurrent_dropout=0.3, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dense(8, activation='gelu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_recurrente_v20_E(forma_entrada, dim_salida, celda='LSTM'):
    """LSTM unidireccional + LayerNorm para ent10/sal05 (kernel cuDNN/Metal activo, hereda de v16_E)."""
    CeldaClase = LSTM if celda == 'LSTM' else GRU
    modelo = Sequential(name=f'{celda}_v20_E')
    modelo.add(CeldaClase(20, activation='tanh', dropout=0.5, input_shape=forma_entrada))
    modelo.add(LayerNormalization())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(16, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

# ==============================================================================
# 4. FAMILIA CONV1D 
# Solo capas Conv1D / SeparableConv1D + pooling + Flatten/GAP.
# ==============================================================================

# --- Versiones Base ---

def construir_conv1d(forma_entrada, dim_salida, filtros=64, kernel=3):
    """2 capas Conv1D + GlobalAveragePooling + Dense. Kernel 3."""
    modelo = Sequential(name='Conv1D')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', input_shape=forma_entrada))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v2(forma_entrada, dim_salida):
    """1 capa Conv1D + Flatten + Dense."""
    modelo = Sequential(name='Conv1D_v2')
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v3(forma_entrada, dim_salida):
    """3 capas Conv1D (64→32→16) + GlobalAveragePooling."""
    modelo = Sequential(name='Conv1D_v3')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    modelo.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v4(forma_entrada, dim_salida):
    """Conv1D dilatada (dilation_rate=2) + GAP."""
    modelo = Sequential(name='Conv1D_v4')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu', dilation_rate=2, padding='causal', input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', dilation_rate=1, padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v5(forma_entrada, dim_salida):
    """SeparableConv1D (depthwise separable)."""
    modelo = Sequential(name='Conv1D_v5')
    modelo.add(SeparableConv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=forma_entrada))
    modelo.add(SeparableConv1D(32, kernel_size=3, activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v6(forma_entrada, dim_salida):
    """Conv1D + MaxPooling1D + Flatten."""
    modelo = Sequential(name='Conv1D_v6')
    modelo.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=forma_entrada))
    modelo.add(MaxPooling1D(pool_size=2, padding='same'))
    modelo.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    modelo.add(Flatten())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo

def construir_conv1d_v7(forma_entrada, dim_salida):
    """Conv1D refinado para ent30/sal30 (heredero de Conv1D_v3). 64->32->16->GAP. gelu, L2 1e-4."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Conv1D_v7')
    modelo.add(Conv1D(64, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg))
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v8(forma_entrada, dim_salida):
    """Conv1D para ent30/sal90. 64->32->16->GAP->Dense 32->dim_salida. Dense head intermedia para target 90d."""
    reg = regularizers.l2(2e-4)
    modelo = Sequential(name='Conv1D_v8')
    modelo.add(Conv1D(64, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Conv1D(32, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg))
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='same', kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(32, activation='gelu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v9(forma_entrada, dim_salida):
    """Conv1D anti-overfit para ent30/sal01. Multi-kernel paralelo (1,3,7) x 6 filtros -> GAP. lr 5e-5, dropout 0.5, L2 5e-4."""
    reg = regularizers.l2(5e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(6, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(6, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama7 = Conv1D(6, kernel_size=7, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama7])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v9')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

# --- Variantes de Emilio (_E) ---

def construir_conv1d_v1_E(forma_entrada, dim_salida, filtros=64, kernel=3):
    """[Emilio] Conv1D dilatada (causal) + MaxPool + Conv1D + GAP."""
    reg = regularizers.l2(1e-4)
    modelo = Sequential(name='Conv1D_v1_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', dilation_rate=2, padding='causal', input_shape=forma_entrada, kernel_regularizer=reg))
    modelo.add(MaxPooling1D(pool_size=3))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu', padding='same', kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo

def construir_conv1d_v2_E(forma_entrada, dim_salida, filtros=32, kernel=7):
    """[Emilio] Conv1D dilatada para sal90."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v2_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', dilation_rate=2, padding='causal', input_shape=forma_entrada, kernel_regularizer=reg))
    modelo.add(MaxPooling1D(pool_size=3))
    modelo.add(Conv1D(filtros // 2, kernel_size=kernel, activation='relu', padding='same', kernel_regularizer=reg))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v3_E(forma_entrada, dim_salida, filtros=16, kernel=7):
    """Conv1D dilatada minimalista para ent90/sal90."""
    modelo = Sequential(name='Conv1D_v3_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', dilation_rate=2, padding='causal', input_shape=forma_entrada))
    modelo.add(MaxPooling1D(pool_size=2, padding='same'))
    modelo.add(Conv1D(max(filtros // 2, 8), kernel_size=kernel, activation='relu', padding='same'))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v4_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo (Inception-like) para ent05/sal01."""
    reg = regularizers.l2(3e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(8, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(8, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(8, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v4_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v5_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel con cabeza Dense para ent05/sal05."""
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
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v6_E(forma_entrada, dim_salida):
    """Conv1D minimalista con kernel = ventana completa para ent05/sal30."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v6_E')
    modelo.add(Conv1D(8, kernel_size=5, activation='gelu', padding='valid', kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v7_E(forma_entrada, dim_salida):
    """Conv1D ultra-minimalista para ent05/sal90."""
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Conv1D_v7_E')
    modelo.add(Conv1D(4, kernel_size=5, activation='gelu', padding='valid', kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v8_E(forma_entrada, dim_salida):
    """Conv1D apilada con dilatacion para ent10/sal01."""
    modelo = Sequential(name='Conv1D_v8_E')
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='causal', input_shape=forma_entrada))
    modelo.add(Conv1D(16, kernel_size=3, activation='gelu', padding='causal', dilation_rate=2))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v9_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel con refinamiento para ent10/sal05."""
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
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v10_E(forma_entrada, dim_salida):
    """Conv1D dilatada minimalista para ent10/sal30."""
    reg = regularizers.l2(5e-4)
    modelo = Sequential(name='Conv1D_v10_E')
    modelo.add(Conv1D(8, kernel_size=5, activation='gelu', padding='causal', dilation_rate=2, kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v11_E(forma_entrada, dim_salida):
    """Conv1D dilatada minimalista para ent10/sal90."""
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Conv1D_v11_E')
    modelo.add(Conv1D(6, kernel_size=5, activation='gelu', padding='valid', dilation_rate=2, kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v12_E(forma_entrada, dim_salida, filtros=32, kernel=7):
    """Conv1D apilada dilatada para ent90/sal30."""
    modelo = Sequential(name='Conv1D_v12_E')
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', padding='causal', input_shape=forma_entrada))
    modelo.add(Conv1D(filtros, kernel_size=kernel, activation='relu', padding='causal', dilation_rate=2))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v13_E(forma_entrada, dim_salida):
    """Conv1D ULTRA-minima single-layer para ent05/sal01."""
    reg = regularizers.l2(1e-3)
    modelo = Sequential(name='Conv1D_v13_E')
    modelo.add(Conv1D(4, kernel_size=5, activation='gelu', padding='valid', kernel_regularizer=reg, input_shape=forma_entrada))
    modelo.add(Flatten())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v13_E_bis(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo reforzado para ent05/sal01 (anti-overfit)."""
    reg = regularizers.l2(5e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(4, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(4, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(4, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v13_E_bis')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v14_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo para ent05/sal05."""
    reg = regularizers.l2(3e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(6, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(6, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(6, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v14_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v15_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo + Dense head para ent05/sal30."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(12, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(12, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(12, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dense(24, activation='gelu', kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v15_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v16_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo + GAP para ent05/sal90."""
    reg = regularizers.l2(2e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(10, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(10, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v16_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v17_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo reforzado para ent10/sal01 (anti-overfit, hereda de v13_E, kernels 1/3/7)."""
    reg = regularizers.l2(5e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(4, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(4, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama7 = Conv1D(4, kernel_size=7, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama7])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v17_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_conv1d_v18_E(forma_entrada, dim_salida):
    """Conv1D multi-kernel paralelo para ent10/sal05 (hereda de v14_E, 8 filtros por rama)."""
    reg = regularizers.l2(3e-4)
    entradas = Input(shape=forma_entrada)
    rama1 = Conv1D(8, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama3 = Conv1D(8, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    rama5 = Conv1D(8, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = Concatenate()([rama1, rama3, rama5])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Conv1D_v18_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

# ==============================================================================
# 5. FAMILIA MIXTA (Arquitectura híbrida: Conv1D + LSTM/GRU + Dense)
# Combinación de capas de distinta naturaleza.
# ==============================================================================

# --- Versiones Base ---

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

def construir_mixto_v2(forma_entrada, dim_salida):
    """Mixto Conv1D + LSTM + Dense para ent30/sal30. Mejora del Mixto base anyadiendo Dense intermedia (LSTM_v4 strategy)."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=3, activation='gelu', padding='same')(entradas)
    x = Conv1D(32, kernel_size=3, activation='gelu', padding='same')(x)
    x = LSTM(32, activation='tanh', dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation='gelu')(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v2')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v3(forma_entrada, dim_salida):
    """Mixto Conv1D + LSTM + Dense para ent30/sal90. Dense head 48 para target 90d."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=3, activation='gelu', padding='same')(entradas)
    x = Conv1D(32, kernel_size=3, activation='gelu', padding='same')(x)
    x = LSTM(32, activation='tanh', dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(48, activation='gelu')(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v3')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v4(forma_entrada, dim_salida):
    """Mixto anti-overfit para ent30/sal01. TimeDistributed(Dense 10) + GRU 10. Mantiene recurrent_dropout=0.3."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(10, activation='tanh'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(10, dropout=0.3)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v4')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

# --- Variantes de Emilio (_E) ---

def construir_mixto_v1_E(forma_entrada, dim_salida):
    """ Conv1D (32 filtros) -> LSTM con dropout interno -> Dense."""
    reg = regularizers.l2(1e-4)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(entradas)
    x = LSTM(64, dropout=0.2, recurrent_dropout=0.1)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v1_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss='mae')
    return modelo

def construir_mixto_v2_E(forma_entrada, dim_salida):
    """ Mixto agresivamente regularizado para ventanas largas (ent90_sal30)."""
    reg = regularizers.l2(1e-3)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=7, activation='relu', padding='causal', kernel_regularizer=reg)(entradas)
    x = Dropout(0.5)(x)
    x = LSTM(32, dropout=0.1)(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v2_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss=keras.losses.Huber(delta=1.0))
    return modelo

def construir_mixto_v3_E(forma_entrada, dim_salida):
    """[Emilio] Mixto Conv1D+LSTM para sal90 (target muy suave)."""
    reg = regularizers.l2(5e-4)
    entradas = Input(shape=forma_entrada)
    x = Conv1D(24, kernel_size=7, activation='relu', padding='causal', kernel_regularizer=reg)(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(32, dropout=0.1)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v3_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v4_E(forma_entrada, dim_salida):
    """Conv1D + LSTM mixto para ent90/sal90, ajustado por random search."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=5, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(64, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v4_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v5_E(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU minimo para ent05/sal01."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(8, activation='gelu'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(8, dropout=0.4)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v5_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v6_E(forma_entrada, dim_salida):
    """Conv multi-kernel + LSTM bidireccional para ent05/sal05."""
    entradas = Input(shape=forma_entrada)
    rama3 = Conv1D(16, kernel_size=3, activation='gelu', padding='same')(entradas)
    rama5 = Conv1D(16, kernel_size=5, activation='gelu', padding='same')(entradas)
    x = Concatenate()([rama3, rama5])
    x = Bidirectional(LSTM(24, dropout=0.3))(x)
    x = Dropout(0.2)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v6_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v7_E(forma_entrada, dim_salida):
    """Conv compacta + GRU diminuta para ent05/sal30."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.4)(x)
    x = GRU(8, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v7_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v8_E(forma_entrada, dim_salida):
    """Conv1D causal + GRU diminuta para ent05/sal90."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(8, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.5)(x)
    x = GRU(4, dropout=0.4)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v8_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v9_E(forma_entrada, dim_salida):
    """Conv1D causal + Bidirectional LSTM para ent10/sal01."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(16, dropout=0.3))(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v9_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v10_E(forma_entrada, dim_salida):
    """Conv multi-kernel + Bidirectional LSTM para ent10/sal05."""
    entradas = Input(shape=forma_entrada)
    rama3 = Conv1D(24, kernel_size=3, activation='gelu', padding='same')(entradas)
    rama5 = Conv1D(24, kernel_size=5, activation='gelu', padding='same')(entradas)
    x = Concatenate()([rama3, rama5])
    x = Bidirectional(LSTM(32, dropout=0.3))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='gelu')(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v10_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v11_E(forma_entrada, dim_salida):
    """Conv1D dilatada + GRU para ent10/sal30."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(16, kernel_size=3, activation='gelu', padding='causal', dilation_rate=2)(entradas)
    x = Dropout(0.4)(x)
    x = GRU(12, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v11_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v12_E(forma_entrada, dim_salida):
    """Conv1D causal + GRU diminuta para ent10/sal90."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(8, kernel_size=3, activation='gelu', padding='causal')(entradas)
    x = Dropout(0.5)(x)
    x = GRU(6, dropout=0.4)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v12_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v13_E(forma_entrada, dim_salida):
    """Conv1D dilatado + LSTM para ent90/sal30."""
    entradas = Input(shape=forma_entrada)
    x = Conv1D(32, kernel_size=7, activation='gelu', padding='causal', dilation_rate=2)(entradas)
    x = Dropout(0.3)(x)
    x = LSTM(32, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v13_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v14_E(forma_entrada, dim_salida):
    """Cross-asset MLP minimo + GRU diminuta para ent05/sal01."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(4, activation='gelu'))(entradas)
    x = Dropout(0.5)(x)
    x = GRU(4, dropout=0.5)(x)
    x = Dropout(0.5)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v14_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v14_E_bis(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU diminuto reforzado para ent05/sal01 (anti-overfit)."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(8, activation='gelu'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(8, dropout=0.5, recurrent_dropout=0.3)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v14_E_bis')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v15_E(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU para ent05/sal05."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(12, activation='gelu'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(12, dropout=0.5)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v15_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v16_E(forma_entrada, dim_salida):
    """Cross-asset MLP + Bidirectional GRU para ent05/sal30."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(24, activation='gelu'))(entradas)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(16, dropout=0.3))(x)
    x = Dense(16, activation='gelu')(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v16_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v17_E(forma_entrada, dim_salida):
    """Cross-asset MLP + BiGRU + Dense head para ent05/sal90."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(20, activation='gelu'))(entradas)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(12, dropout=0.3))(x)
    x = Dense(24, activation='gelu')(x)
    x = Dropout(0.3)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v17_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v18_E(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU diminuto para ent10/sal01 (anti-overfit, hereda de v14_E, mantiene recurrent_dropout)."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(8, activation='gelu'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(10, dropout=0.5, recurrent_dropout=0.3)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v18_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss='mae')
    return modelo

def construir_mixto_v19_E(forma_entrada, dim_salida):
    """Cross-asset MLP + GRU para ent10/sal05 (kernel cuDNN/Metal activo, hereda de v15_E)."""
    entradas = Input(shape=forma_entrada)
    x = TimeDistributed(Dense(16, activation='gelu'))(entradas)
    x = Dropout(0.4)(x)
    x = GRU(16, dropout=0.5)(x)
    x = Dropout(0.4)(x)
    salidas = Dense(dim_salida)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_v19_E')
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mae')
    return modelo
