import inspect
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utilidades import modelos
from utilidades.carga_datos import aplanar_X


FIT_KEYS = {"epochs", "batch_size", "verbose"}
CB_KEYS = {
    "factor", "rlr_patience", "min_delta", "min_lr",
    "es_patience", "restore_best_weights", "rlr_monitor", "es_monitor",
    "track_train_eval", "train_eval_samples",
    "gap_weight", "gap_metric", "gap_target", "gap_excess_weight",
}


def _build_model_from_modelos(model_fn_name, cfg, X_train, y_train):
    fn = getattr(modelos, model_fn_name)
    sig = inspect.signature(fn)
    pnames = set(sig.parameters.keys())

    # Decide si el modelo espera entrada plana (Dense) o secuencia (RNN/Conv1D)
    if "dim_entrada" in pnames:
        Xtr = aplanar_X(X_train) if X_train.ndim == 3 else X_train
        base_args = {"dim_entrada": Xtr.shape[1], "dim_salida": y_train.shape[1]}
    elif "forma_entrada" in pnames:
        Xtr = X_train
        base_args = {"forma_entrada": Xtr.shape[1:], "dim_salida": y_train.shape[1]}
    else:
        raise ValueError(f"No reconozco firma de {model_fn_name}: {sig}")

    # kwargs de constructor que sí existen en ese método
    ctor_args = {}
    for k in pnames:
        if k in base_args:
            continue
        if k in cfg:
            ctor_args[k] = cfg[k]

    model = fn(**base_args, **ctor_args)

    # Permite tunear lr/clipnorm sin editar modelos.py (recompile en runtime)
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

    # Val con mismo tipo de entrada
    if "dim_entrada" in inspect.signature(getattr(modelos, model_fn_name)).parameters:
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
    # parámetros realmente tuneables para ese método
    sig = inspect.signature(getattr(modelos, model_fn_name))
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
    # Prueba todas las combinaciones posibles de search_steps.
    sig = inspect.signature(getattr(modelos, model_fn_name))
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


def plot_best_history(hist, title="Convergencia — Mejor configuración"):
    score_key = "balanced_score" if "balanced_score" in hist.history else "val_loss"
    best_epoch = int(np.argmin(hist.history[score_key])) + 1
    train_key = "train_eval_loss" if "train_eval_loss" in hist.history else "loss"
    train_label = "Train (eval)" if train_key == "train_eval_loss" else "Train"
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history[train_key], label=train_label)
    plt.plot(hist.history["val_loss"], label="Validación")
    plt.axvline(best_epoch - 1, color="gray", ls="--", alpha=0.6, label=f"Mejor época: {best_epoch}")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def successive_halving_search(model_fn_name, base_cfg, search_steps, X_train, y_train,
                              X_val, y_val, n_candidates=24, keep_top=6,
                              short_epochs=12, final_epochs=60, progress_every=None):
    # Ronda barata con pocas epocas; ronda final solo con los mejores candidatos.
    # No usa azar: selecciona candidatos repartidos uniformemente en la grilla.
    sig = inspect.signature(getattr(modelos, model_fn_name))
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
