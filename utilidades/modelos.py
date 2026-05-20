import inspect
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utilidades.carga_datos import aplanar_X
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

# -----------------------------------------------------------------------------
# MODELOS PARAMETRIZABLES Y UTILIDADES DE TUNING
# -----------------------------------------------------------------------------

FIT_KEYS = {"epochs", "batch_size", "verbose"}
CB_KEYS = {
    "factor", "rlr_patience", "min_delta", "min_lr",
    "es_patience", "restore_best_weights", "rlr_monitor", "es_monitor",
    "track_train_eval", "train_eval_samples",
    "gap_weight", "gap_metric", "gap_target", "gap_excess_weight",
}


def construir_dense_v2_param(
    dim_entrada,
    dim_salida,
    units=(128, 64, 32),
    dropout=0.3,
    activation='relu',
    l2=0.0,
):
    """Version parametrizable de Dense_v2 para busqueda de arquitectura."""
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    modelo = Sequential(name='Dense_v2_param')
    for i, n_units in enumerate(units):
        kwargs = {"activation": activation, "kernel_regularizer": reg}
        if i == 0:
            kwargs["input_shape"] = (dim_entrada,)
        modelo.add(Dense(n_units, **kwargs))
        if dropout and dropout > 0:
            modelo.add(Dropout(dropout))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo



def construir_dense_param(
    dim_entrada,
    dim_salida,
    units=(256, 128),
    dropout=0.2,
    activation='relu',
    l2=0.0,
):
    """Version parametrizable equivalente al modelo Dense base."""
    return construir_dense_v2_param(dim_entrada, dim_salida, units, dropout, activation, l2)


def construir_dense_v3_param(
    dim_entrada,
    dim_salida,
    units=(512, 256, 128, 64),
    dropout=0.1,
    activation='relu',
    l2=0.0,
):
    """Version parametrizable equivalente a Dense_v3."""
    return construir_dense_v2_param(dim_entrada, dim_salida, units, dropout, activation, l2)

def construir_dense_v4_param(
    dim_entrada,
    dim_salida,
    units=(128, 64),
    dropout=0.2,
    activation='relu',
    l2=1e-4,
):
    """Version parametrizable de Dense_v4 con L2 configurable."""
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    modelo = Sequential(name='Dense_v4_param')
    for i, n_units in enumerate(units):
        kwargs = {'activation': activation, 'kernel_regularizer': reg}
        if i == 0:
            kwargs['input_shape'] = (dim_entrada,)
        modelo.add(Dense(n_units, **kwargs))
        if dropout and dropout > 0:
            modelo.add(Dropout(dropout))
    modelo.add(Dense(dim_salida))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo



def construir_dense_v5_param(
    dim_entrada,
    dim_salida,
    units=(256,),
    dropout=0.2,
    activation='tanh',
    l2=0.0,
):
    """Version parametrizable equivalente a Dense_v5."""
    return construir_dense_v2_param(dim_entrada, dim_salida, units, dropout, activation, l2)


def construir_dense_v6_param(
    dim_entrada,
    dim_salida,
    units=(32, 64, 128),
    dropout=0.2,
    activation='elu',
    l2=0.0,
):
    """Version parametrizable equivalente a Dense_v6."""
    return construir_dense_v2_param(dim_entrada, dim_salida, units, dropout, activation, l2)

def construir_lstm_param(
    forma_entrada,
    dim_salida,
    units=16,
    dropout=0.35,
    recurrent_dropout=0.0,
    l2=5e-5,
):
    """LSTM parametrizable para busquedas y entrenamientos balanceados."""
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    modelo = Sequential(name='LSTM_param')
    modelo.add(LSTM(
        units=units,
        input_shape=forma_entrada,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg,
        return_sequences=False,
    ))
    modelo.add(Dense(dim_salida, kernel_regularizer=reg))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_conv1d_param(
    forma_entrada,
    dim_salida,
    filtros=24,
    kernel=3,
    dropout=0.40,
    l2=1e-4,
):
    """Conv1D parametrizable para ajustar filtros, kernel y regularizacion."""
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    modelo = Sequential(name='Conv1D_param')
    modelo.add(Conv1D(
        filtros,
        kernel_size=kernel,
        activation='relu',
        padding='same',
        kernel_regularizer=reg,
        input_shape=forma_entrada,
    ))
    modelo.add(Conv1D(
        max(8, filtros // 2),
        kernel_size=kernel,
        activation='relu',
        padding='same',
        kernel_regularizer=reg,
    ))
    modelo.add(GlobalAveragePooling1D())
    modelo.add(Dropout(dropout))
    modelo.add(Dense(dim_salida, kernel_regularizer=reg))
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def construir_mixto_param(
    forma_entrada,
    dim_salida,
    filtros=24,
    lstm_units=16,
    kernel=3,
    dropout=0.40,
    l2=1e-4,
):
    """Arquitectura Conv1D + LSTM parametrizable."""
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    entradas = Input(shape=forma_entrada)
    x = Conv1D(
        filtros,
        kernel_size=kernel,
        activation='relu',
        padding='same',
        kernel_regularizer=reg,
    )(entradas)
    x = Dropout(dropout)(x)
    x = LSTM(
        lstm_units,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg,
    )(x)
    x = Dropout(dropout)(x)
    salidas = Dense(dim_salida, kernel_regularizer=reg)(x)
    modelo = Model(inputs=entradas, outputs=salidas, name='Mixto_param')
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def _get_model_fn(model_fn_name):
    return globals()[model_fn_name]


def _build_model_from_modelos(model_fn_name, cfg, X_train, y_train):
    fn = _get_model_fn(model_fn_name)
    sig = inspect.signature(fn)
    pnames = set(sig.parameters.keys())

    if "dim_entrada" in pnames:
        Xtr = aplanar_X(X_train) if X_train.ndim == 3 else X_train
        base_args = {"dim_entrada": Xtr.shape[1], "dim_salida": y_train.shape[1]}
    elif "forma_entrada" in pnames:
        Xtr = X_train
        base_args = {"forma_entrada": Xtr.shape[1:], "dim_salida": y_train.shape[1]}
    else:
        raise ValueError(f"No reconozco firma de {model_fn_name}: {sig}")

    ctor_args = {}
    for k in pnames:
        if k in base_args:
            continue
        if k in cfg:
            ctor_args[k] = cfg[k]

    model = fn(**base_args, **ctor_args)

    if "lr" in cfg or "clipnorm" in cfg:
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=cfg.get("lr", 1e-3),
                clipnorm=cfg.get("clipnorm"),
            ),
            loss="mae"
        )

    return model, Xtr


class TrainEvalLoss(keras.callbacks.Callback):
    def __init__(self, X_train_eval, y_train_eval, batch_size=64,
                 gap_weight=0.0, gap_target=0.0, gap_excess_weight=0.0):
        super().__init__()
        self.X_train_eval = X_train_eval
        self.y_train_eval = y_train_eval
        self.batch_size = batch_size
        self.gap_weight = gap_weight
        self.gap_target = gap_target
        self.gap_excess_weight = gap_excess_weight

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = self.model.evaluate(
            self.X_train_eval,
            self.y_train_eval,
            batch_size=self.batch_size,
            verbose=0,
        )
        logs["train_eval_loss"] = float(value)
        if "val_loss" in logs:
            gap = abs(float(value) - float(logs["val_loss"]))
            excess_gap = max(0.0, gap - self.gap_target)
            logs["train_val_gap"] = gap
            logs["balanced_score"] = (
                float(logs["val_loss"])
                + self.gap_weight * gap
                + self.gap_excess_weight * excess_gap
            )


def _make_callbacks(cfg, X_train_eval=None, y_train_eval=None):
    callbacks = []
    if cfg.get("track_train_eval") and X_train_eval is not None and y_train_eval is not None:
        n_eval = cfg.get("train_eval_samples")
        if n_eval:
            X_train_eval = X_train_eval[-n_eval:]
            y_train_eval = y_train_eval[-n_eval:]
        callbacks.append(
            TrainEvalLoss(
                X_train_eval,
                y_train_eval,
                batch_size=cfg.get("batch_size", 64),
                gap_weight=cfg.get("gap_weight", 0.0),
                gap_target=cfg.get("gap_target", 0.0),
                gap_excess_weight=cfg.get("gap_excess_weight", 0.0),
            )
        )

    callbacks.append(
        ReduceLROnPlateau(
            monitor=cfg.get("rlr_monitor", "val_loss"),
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("rlr_patience", 3),
            min_delta=cfg.get("min_delta", 1e-5),
            min_lr=cfg.get("min_lr", 1e-8),
            mode="min",
            verbose=0,
        )
    )
    if cfg.get("es_patience"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.get("es_monitor", "val_loss"),
                patience=cfg["es_patience"],
                min_delta=cfg.get("min_delta", 1e-5),
                restore_best_weights=cfg.get("restore_best_weights", True),
                mode="min",
                verbose=0,
            )
        )
    return callbacks


def _score_history(hist, cfg):
    score_key = cfg.get("es_monitor")
    if score_key in hist.history:
        best_idx = int(np.argmin(hist.history[score_key]))
    else:
        best_idx = int(np.argmin(hist.history["val_loss"]))
    best_val = float(hist.history["val_loss"][best_idx])

    gap_key = cfg.get("gap_metric")
    if gap_key is None:
        gap_key = "train_eval_loss" if "train_eval_loss" in hist.history else "loss"

    train_value = float(hist.history[gap_key][best_idx]) if gap_key in hist.history else np.nan
    gap = abs(train_value - best_val) if np.isfinite(train_value) else np.nan
    if "balanced_score" in hist.history:
        score = float(hist.history["balanced_score"][best_idx])
    else:
        score = best_val + cfg.get("gap_weight", 0.0) * (gap if np.isfinite(gap) else 0.0)
    return score, best_val, best_idx + 1, train_value, gap


def _train_one(model_fn_name, cfg, X_train, y_train, X_val, y_val):
    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg.get("seed", 42))

    model, Xtr = _build_model_from_modelos(model_fn_name, cfg, X_train, y_train)

    if "dim_entrada" in inspect.signature(_get_model_fn(model_fn_name)).parameters:
        Xv = aplanar_X(X_val) if X_val.ndim == 3 else X_val
    else:
        Xv = X_val

    hist = model.fit(
        Xtr, y_train,
        validation_data=(Xv, y_val),
        epochs=cfg.get("epochs", 50),
        batch_size=cfg.get("batch_size", 64),
        callbacks=_make_callbacks(cfg, Xtr, y_train),
        verbose=cfg.get("verbose", 0),
    )

    score, best_val, best_epoch, train_value, gap = _score_history(hist, cfg)
    return score, best_val, best_epoch, train_value, gap, hist, model


def ofat_search(model_fn_name, base_cfg, search_steps, X_train, y_train, X_val, y_val):
    sig = inspect.signature(_get_model_fn(model_fn_name))
    ctor_tuneables = set(sig.parameters.keys()) - {"dim_entrada", "dim_salida", "forma_entrada"}
    valid = ctor_tuneables | FIT_KEYS | CB_KEYS | {"lr", "seed", "clipnorm"}

    trials = []
    best_cfg = base_cfg.copy()

    best_score, best_val, best_epoch, best_train, best_gap, best_hist, best_model = _train_one(
        model_fn_name, best_cfg, X_train, y_train, X_val, y_val
    )
    trials.append({
        "param": "baseline", "value": "-", "score": best_score,
        "best_val_loss": best_val, "best_train_loss": best_train,
        "train_val_gap": best_gap, "best_epoch": best_epoch, **best_cfg
    })

    for param, values in search_steps:
        if param not in valid:
            raise ValueError(
                f"'{param}' no es tuneable para {model_fn_name}. "
                f"Tuneables: {sorted(valid)}"
            )

        local_best = (best_score, best_val, best_epoch, best_train, best_gap, best_hist, best_model, best_cfg.copy())

        for v in values:
            cfg = best_cfg.copy()
            cfg[param] = v

            score, val, ep, train_value, gap, hist, model = _train_one(
                model_fn_name, cfg, X_train, y_train, X_val, y_val
            )

            trials.append({
                "param": param, "value": v, "score": score,
                "best_val_loss": val, "best_train_loss": train_value,
                "train_val_gap": gap, "best_epoch": ep, **cfg
            })

            if score < local_best[0]:
                local_best = (score, val, ep, train_value, gap, hist, model, cfg.copy())

        best_score, best_val, best_epoch, best_train, best_gap, best_hist, best_model, best_cfg = local_best
        print(
            f"{param:12} -> mejor: {best_cfg[param]} | "
            f"score={best_score:.6f} | val_loss={best_val:.6f} | gap={best_gap:.6f} "
            f"(epoca {best_epoch})"
        )

    df_trials = pd.DataFrame(trials).sort_values("score").reset_index(drop=True)
    return best_cfg, best_val, best_epoch, best_hist, best_model, df_trials


def grid_search(model_fn_name, base_cfg, search_steps, X_train, y_train, X_val, y_val,
                progress_every=25):
    sig = inspect.signature(_get_model_fn(model_fn_name))
    ctor_tuneables = set(sig.parameters.keys()) - {"dim_entrada", "dim_salida", "forma_entrada"}
    valid = ctor_tuneables | FIT_KEYS | CB_KEYS | {"lr", "seed", "clipnorm"}

    for param, _ in search_steps:
        if param not in valid:
            raise ValueError(
                f"'{param}' no es tuneable para {model_fn_name}. "
                f"Tuneables: {sorted(valid)}"
            )

    params = [param for param, _ in search_steps]
    values_grid = [values for _, values in search_steps]
    total = int(np.prod([len(values) for values in values_grid])) if values_grid else 1

    trials = []
    best_score = np.inf
    best_val = np.inf
    best_epoch = None
    best_hist = None
    best_model = None
    best_cfg = None

    for i, combo in enumerate(product(*values_grid), start=1):
        cfg = base_cfg.copy()
        cfg.update(dict(zip(params, combo)))

        score, val, ep, train_value, gap, hist, model = _train_one(
            model_fn_name, cfg, X_train, y_train, X_val, y_val
        )

        trials.append({
            "combo": i,
            "score": score,
            "best_val_loss": val,
            "best_train_loss": train_value,
            "train_val_gap": gap,
            "best_epoch": ep,
            **cfg
        })

        if score < best_score:
            best_score = score
            best_val = val
            best_epoch = ep
            best_hist = hist
            best_model = model
            best_cfg = cfg.copy()

        if progress_every and (i == 1 or i % progress_every == 0 or i == total):
            print(
                f"{model_fn_name}: {i}/{total} combinaciones | "
                f"mejor score={best_score:.6f} | val_loss={best_val:.6f}"
            )

    df_trials = pd.DataFrame(trials).sort_values("score").reset_index(drop=True)
    return best_cfg, best_val, best_epoch, best_hist, best_model, df_trials


def plot_best_history(hist, title="Convergencia - Mejor configuracion"):
    score_key = "balanced_score" if "balanced_score" in hist.history else "val_loss"
    best_epoch = int(np.argmin(hist.history[score_key])) + 1
    train_key = "train_eval_loss" if "train_eval_loss" in hist.history else "loss"
    train_label = "Train (eval)" if train_key == "train_eval_loss" else "Train"
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history[train_key], label=train_label)
    plt.plot(hist.history["val_loss"], label="Validacion")
    plt.axvline(best_epoch - 1, color="gray", ls="--", alpha=0.6, label=f"Mejor epoca: {best_epoch}")
    plt.title(title)
    plt.xlabel("Epoca")
    plt.ylabel("MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def successive_halving_search(model_fn_name, base_cfg, search_steps, X_train, y_train,
                              X_val, y_val, n_candidates=24, keep_top=6,
                              short_epochs=12, final_epochs=60, progress_every=None):
    sig = inspect.signature(_get_model_fn(model_fn_name))
    ctor_tuneables = set(sig.parameters.keys()) - {"dim_entrada", "dim_salida", "forma_entrada"}
    valid = ctor_tuneables | FIT_KEYS | CB_KEYS | {"lr", "seed", "clipnorm"}

    for param, _ in search_steps:
        if param not in valid:
            raise ValueError(
                f"'{param}' no es tuneable para {model_fn_name}. "
                f"Tuneables: {sorted(valid)}"
            )

    params = [param for param, _ in search_steps]
    values_grid = [list(values) for _, values in search_steps]
    grid_sizes = [len(values) for values in values_grid]
    total = int(np.prod(grid_sizes)) if grid_sizes else 1
    n_candidates = min(n_candidates, total)
    keep_top = min(keep_top, n_candidates)

    selected_idx = np.linspace(0, total - 1, n_candidates, dtype=int)

    def combo_from_index(index):
        combo = []
        for values in reversed(values_grid):
            index, value_idx = divmod(int(index), len(values))
            combo.append(values[value_idx])
        return tuple(reversed(combo))

    combos = [combo_from_index(i) for i in selected_idx]
    progress_every = progress_every or max(1, n_candidates // 4)

    short_trials = []
    print(f"Ronda corta: {n_candidates}/{total} candidatos con {short_epochs} epocas")
    for i, combo in enumerate(combos, start=1):
        cfg = base_cfg.copy()
        cfg.update(dict(zip(params, combo)))
        cfg["epochs"] = short_epochs

        score, val, ep, train_value, gap, hist, model = _train_one(
            model_fn_name, cfg, X_train, y_train, X_val, y_val
        )

        short_trials.append({
            "trial": i,
            "score": score,
            "best_val_loss": val,
            "best_train_loss": train_value,
            "train_val_gap": gap,
            "best_epoch": ep,
            **cfg
        })

        if progress_every and (i == 1 or i % progress_every == 0 or i == n_candidates):
            best_short = min(row["score"] for row in short_trials)
            print(
                f"{model_fn_name}: {i}/{n_candidates} candidatos cortos | "
                f"mejor score={best_short:.6f}"
            )

    df_short = pd.DataFrame(short_trials).sort_values("score").reset_index(drop=True)

    finalists = df_short.head(keep_top).to_dict("records")
    trials = []
    best_score = np.inf
    best_val = np.inf
    best_epoch = None
    best_hist = None
    best_model = None
    best_cfg = None

    print(f"Ronda final: {len(finalists)} candidatos con {final_epochs} epocas")
    for i, row in enumerate(finalists, start=1):
        cfg = base_cfg.copy()
        for param, _ in search_steps:
            cfg[param] = row[param]
        cfg["epochs"] = final_epochs

        score, val, ep, train_value, gap, hist, model = _train_one(
            model_fn_name, cfg, X_train, y_train, X_val, y_val
        )

        trials.append({
            "trial": i,
            "score": score,
            "best_val_loss": val,
            "best_train_loss": train_value,
            "train_val_gap": gap,
            "best_epoch": ep,
            **cfg
        })

        if score < best_score:
            best_score = score
            best_val = val
            best_epoch = ep
            best_hist = hist
            best_model = model
            best_cfg = cfg.copy()

        print(f"final {i}/{len(finalists)} | mejor score={best_score:.6f} | val_loss={best_val:.6f}")

    df_final = pd.DataFrame(trials).sort_values("score").reset_index(drop=True)
    return best_cfg, best_val, best_epoch, best_hist, best_model, df_short, df_final

