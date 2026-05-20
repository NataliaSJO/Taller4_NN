import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=FutureWarning)

TICKERS = [
    'AEP', 'BA', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'GD', 'GE',
    'HON', 'HPQ', 'IBM', 'IP', 'JNJ', 'KO', 'KR', 'MMM', 'MO', 'MRK',
    'MSI', 'PG', 'XOM'
]
FECHA_INICIO = '1945-01-01'
SEMILLA = 42


def cargar_retornos(ruta_cache='../resultados/retornos_cache.csv'):
    """Descarga precios via yfinance, calcula log-retornos y cachea en CSV."""
    if ruta_cache and os.path.exists(ruta_cache):
        return pd.read_csv(ruta_cache, index_col=0, parse_dates=True)

    import yfinance as yf
    precios = yf.download(TICKERS, start=FECHA_INICIO, auto_adjust=True, progress=True)['Close']
    precios.dropna(axis=1, inplace=True)
    retornos = np.log(precios).diff().dropna()

    if ruta_cache:
        os.makedirs(os.path.dirname(ruta_cache), exist_ok=True)
        retornos.to_csv(ruta_cache)

    return retornos


def create_time_series_data(data, input_window_size, output_window_size):
    """
    Genera secuencias de entrada y promedios de salida para datos de series temporales.

    Args:
        data (pd.DataFrame o np.array): Los datos de la serie temporal.
        input_window_size (int): El número de pasos de tiempo para la secuencia de entrada (X).
        output_window_size (int): El número de pasos de tiempo para calcular el promedio de la salida (y).

    Returns:
        tuple: (X, y) donde X son las secuencias de entrada y y son los promedios de salida.
               X tendrá la forma (num_samples, input_window_size, num_features).
               y tendrá la forma (num_samples, num_features) si output_window_size > 0.
               Si output_window_size == 0, y contendrá el último valor de la ventana de entrada.
    """
    X, y = [], []
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    num_features = data_array.shape[1]

    for i in range(len(data_array) - input_window_size - output_window_size + 1):
        input_sequence = data_array[i: i + input_window_size]
        X.append(input_sequence)

        if output_window_size > 0:
            output_sequence = data_array[i + input_window_size: i + input_window_size + output_window_size]
            average_output = np.mean(output_sequence, axis=0)
            y.append(average_output)
        else:
            y.append(data_array[i + input_window_size - 1])

    return np.array(X), np.array(y)


def dividir_datos(X, y, test=0.10, val=0.05, semilla=SEMILLA):
    """
    Divide en train/val/test manteniendo orden cronológico (shuffle=False).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test, shuffle=False, random_state=semilla
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val, shuffle=False, random_state=semilla
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def aplanar_X(X_seq):
    """Reshape (N, V, 23) → (N, V*23) para modelos Dense."""
    return X_seq.reshape(X_seq.shape[0], -1)
