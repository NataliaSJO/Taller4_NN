import os
import json
import numpy as np
import pandas as pd


def calcular_mae(y_real, y_pred):
    """MAE medio sobre todas las muestras y activos."""
    return float(np.mean(np.abs(y_real - y_pred)))


def evaluar_modelo(modelo, X_tr, y_tr, X_val, y_val, X_ts, y_ts,
                   nombre, es_plano=False):
    """
    Predice sobre train/val/test y devuelve un dict con las métricas.

    Args:
        es_plano: True para modelos Dense que esperan entrada 2D aplanada.
    Returns:
        dict con claves: modelo, mae_train, mae_val, mae_test, n_params
    """
    pred_tr  = modelo.predict(X_tr,  verbose=0)
    pred_val = modelo.predict(X_val, verbose=0)
    pred_ts  = modelo.predict(X_ts,  verbose=0)

    return {
        'modelo':    nombre,
        'mae_train': calcular_mae(y_tr,  pred_tr),
        'mae_val':   calcular_mae(y_val, pred_val),
        'mae_test':  calcular_mae(y_ts,  pred_ts),
        'n_params':  int(modelo.count_params()),
    }


def evaluar_sklearn(modelo, X_tr, y_tr, X_val, y_val, X_ts, y_ts, nombre):
    """Igual que evaluar_modelo pero para modelos sklearn (sin count_params)."""
    pred_tr  = modelo.predict(X_tr)
    pred_val = modelo.predict(X_val)
    pred_ts  = modelo.predict(X_ts)

    return {
        'modelo':    nombre,
        'mae_train': calcular_mae(y_tr,  pred_tr),
        'mae_val':   calcular_mae(y_val, pred_val),
        'mae_test':  calcular_mae(y_ts,  pred_ts),
        'n_params':  0,
    }


def evaluar_buyhold(y_tr, y_val, y_ts):
    """Baseline Buy-and-Hold: predice siempre la media del entrenamiento."""
    media = np.mean(y_tr, axis=0, keepdims=True)
    pred_tr  = np.tile(media, (len(y_tr),  1))
    pred_val = np.tile(media, (len(y_val), 1))
    pred_ts  = np.tile(media, (len(y_ts),  1))

    return {
        'modelo':    'BuyAndHold',
        'mae_train': calcular_mae(y_tr,  pred_tr),
        'mae_val':   calcular_mae(y_val, pred_val),
        'mae_test':  calcular_mae(y_ts,  pred_ts),
        'n_params':  0,
    }


def guardar_resultados(resultados, ventana_entrada, ventana_salida,
                        seccion='competicion',
                        carpeta='../resultados/metricas/'):
    """
    Escribe los resultados en resultados/metricas/entXX_salYY.json.

    Cada entrada lleva el campo 'seccion' ('competicion' o 'investigacion')
    para que los cuadernos de analisis puedan filtrar correctamente.
    Los resultados de la misma seccion se sobreescriben; los de la otra
    seccion se conservan.
    """
    os.makedirs(carpeta, exist_ok=True)
    nombre = f'ent{ventana_entrada:02d}_sal{ventana_salida:02d}.json'
    ruta = os.path.join(carpeta, nombre)

    # Marcar seccion en cada resultado
    nuevos = [{**r, 'seccion': seccion} for r in resultados]

    # Conservar entradas de la otra seccion si el fichero ya existe
    existentes = []
    if os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as f:
            existentes = json.load(f)
        existentes = [e for e in existentes if e.get('seccion') != seccion]

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(existentes + nuevos, f, indent=2, ensure_ascii=False)
    print(f'Resultados [{seccion}] guardados en: {ruta}')


def cargar_todos_resultados(carpeta='../resultados/metricas/',
                             seccion=None):
    """
    Lee todos los JSON de la carpeta y devuelve un DataFrame.

    Args:
        seccion: 'competicion', 'investigacion' o None (devuelve todo).

    Columnas: ventana_entrada, ventana_salida, seccion, modelo,
              mae_train, mae_val, mae_test, n_params
    """
    filas = []
    if not os.path.exists(carpeta):
        print(f'Carpeta no encontrada: {carpeta}')
        return pd.DataFrame()

    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith('.json')])
    if not archivos:
        print('No se encontraron ficheros JSON en:', carpeta)
        return pd.DataFrame()

    for archivo in archivos:
        # Nombre esperado: ent05_sal01.json
        partes = archivo.replace('.json', '').split('_')
        v_ent = int(partes[0].replace('ent', ''))
        v_sal = int(partes[1].replace('sal', ''))
        with open(os.path.join(carpeta, archivo), encoding='utf-8') as f:
            datos = json.load(f)
        for entrada in datos:
            if seccion and entrada.get('seccion') != seccion:
                continue
            fila = {'ventana_entrada': v_ent, 'ventana_salida': v_sal}
            fila.update(entrada)
            filas.append(fila)

    return pd.DataFrame(filas)
