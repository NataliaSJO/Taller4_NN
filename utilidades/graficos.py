import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def graficar_convergencia(historia, nombre_modelo, ax=None):
    """Curvas de pérdida train/val del objeto history de Keras."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historia.history['loss'],     label='Train')
    ax.plot(historia.history['val_loss'], label='Validación')
    ax.set_title(f'Convergencia — {nombre_modelo}')
    ax.set_xlabel('Época')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def graficar_barras_mae(resultados, ventana_entrada, ventana_salida,
                         ruta_guardado=None):
    """
    Gráfico de barras agrupadas (train/val/test) por modelo.
    Uno de los 16 gráficos requeridos.
    """
    df = pd.DataFrame(resultados)
    modelos = df['modelo'].tolist()
    x = np.arange(len(modelos))
    ancho = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(modelos) * 1.5), 5))
    ax.bar(x - ancho, df['mae_train'], ancho, label='Train',      color='steelblue')
    ax.bar(x,         df['mae_val'],   ancho, label='Validación',  color='orange')
    ax.bar(x + ancho, df['mae_test'],  ancho, label='Test',        color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title(f'MAE por modelo — Entrada {ventana_entrada}d / Salida {ventana_salida}d')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=150)
    plt.show()


def graficar_resumen(df, ventana_salida, ruta_guardado=None):
    """
    Gráfico resumen para una ventana de salida dada.
    Muestra todas las ventanas de entrada, agrupadas por modelo.
    """
    datos = df[df['ventana_salida'] == ventana_salida].copy()
    if datos.empty:
        print(f'Sin datos para ventana_salida={ventana_salida}')
        return

    modelos = datos['modelo'].unique()
    ventanas_entrada = sorted(datos['ventana_entrada'].unique())
    x = np.arange(len(modelos))
    ancho = 0.8 / len(ventanas_entrada)

    fig, ax = plt.subplots(figsize=(max(10, len(modelos) * 2), 5))
    for i, v_ent in enumerate(ventanas_entrada):
        subset = datos[datos['ventana_entrada'] == v_ent]
        maes = [subset[subset['modelo'] == m]['mae_test'].values[0]
                if m in subset['modelo'].values else np.nan
                for m in modelos]
        offset = (i - len(ventanas_entrada) / 2 + 0.5) * ancho
        ax.bar(x + offset, maes, ancho, label=f'Entrada {v_ent}d')

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha='right')
    ax.set_ylabel('MAE Test')
    ax.set_title(f'Resumen — Ventana de salida {ventana_salida}d')
    ax.legend(title='Ventana entrada')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=150)
    plt.show()


def graficar_matriz_competicion(df, metrica='mae_test', ruta_guardado=None):
    """
    Heatmap 4×4 con el mejor MAE test por combinación de ventanas.
    """
    ventanas_entrada = sorted(df['ventana_entrada'].unique())
    ventanas_salida  = sorted(df['ventana_salida'].unique())

    matriz = np.full((len(ventanas_entrada), len(ventanas_salida)), np.nan)
    for i, v_ent in enumerate(ventanas_entrada):
        for j, v_sal in enumerate(ventanas_salida):
            subset = df[(df['ventana_entrada'] == v_ent) &
                        (df['ventana_salida']  == v_sal)]
            if not subset.empty:
                matriz[i, j] = subset[metrica].min()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap='viridis_r', aspect='auto')
    plt.colorbar(im, ax=ax, label=metrica)

    ax.set_xticks(range(len(ventanas_salida)))
    ax.set_yticks(range(len(ventanas_entrada)))
    ax.set_xticklabels([f'{v}d' for v in ventanas_salida])
    ax.set_yticklabels([f'{v}d' for v in ventanas_entrada])
    ax.set_xlabel('Ventana de salida')
    ax.set_ylabel('Ventana de entrada')
    ax.set_title(f'Matriz de competición — mejor {metrica}')

    for i in range(len(ventanas_entrada)):
        for j in range(len(ventanas_salida)):
            if not np.isnan(matriz[i, j]):
                ax.text(j, i, f'{matriz[i, j]:.4f}', ha='center', va='center',
                        color='white', fontsize=8)

    plt.tight_layout()
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=150)
    plt.show()
