import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

files_per_class = {
    0: ["Sebas-A-musc-3-ABC-L01.txt", "Lais-A-musc-3-ABC-L01.txt", "Alejandra-A-Musc-ABC-L01.txt", "Dani_A_1-L01.txt", "Dani_A_2-L01.txt", "Dani_A_3-L01.txt", "Stephany-A-Musc-ABC-L01.txt", "AnaPao-A-Musc-ABC-L01.txt",  "Isa3musculosA-L01.txt", "Diego-A-3abc-L01.txt", "Danna3musculos_A-L01.txt", "Sara-A-Musc-ABC-L01.txt", "Abril_A-L01.txt", "Felipe_A-L01.txt"],  # Clase 0: Seña A
    1: ["Sebas-B-musc-3-ABC-L01.txt", "Lais-B-Musc-3-ABC-L01.txt", "Alejandra-B-Musc-ABC-L01.txt", "Dani_B_1-L01.txt", "Dani_B_2-L01.txt", "Dani_B_3-L01.txt", "Stephany-B-Musc-ABC-L01.txt", "AnaPao-B-Musc-ABC-L01.txt",  "Isa3musculosB-L01.txt", "Diego-B-Musc-ABC-L01.txt", "Danna3Musculos_B-L01.txt", "Sara-B-Musc-ABC-L01.txt", "Abril_B-L01.txt", "Felipe_B-L01.txt"],  # Clase 1: Seña B
    2: ["Sebas-C-Musc-3-ABC-L01.txt", "Lais-C-Musc-3-ABC-L01.txt", "Alejandra-C-Musc-ABC-L01.txt", "Dani_C_1-L01.txt", "Dani_C_2-L01.txt", "Dani_C_3-L01.txt", "Stephany-C-Musc-ABC-L01.txt", "AnaPao-C-Musc-ABC-L01.txt",  "Isa3musculosC-L01.txt", "Diego-C-Musc-ABC-L01.txt", "Danna3musculos_C-L01.txt", "Sara-C-Musc-ABC-L01.txt", "Abril_C-L01.txt", "Felipe_C-L01.txt"],  # Clase 2: Seña C
}

fs = 2000  # Frecuencia de muestreo real (2000 Hz)


# 2. FUNCIONES DE PREPROCESAMIENTO Y FILTRADO


def bandpass_filter(data, fs=2000, lowcut=20, highcut=450, order=4):
    """Aplica un filtro pasabanda Butterworth para limpiar ruido en la señal EMG."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def rectify_signal(data):
    """Rectifica la señal EMG aplicando valor absoluto."""
    return np.abs(data)



# 3. PROCESAMIENTO Y EXTRACCIÓN DE CARACTERÍSTICAS (FEATURES)



def process_files_for_class(file_paths_for_class, label):
    combined_df = pd.DataFrame()

    # Cargar y concatenar archivos de la misma clase
    for file_path in file_paths_for_class:
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo '{file_path}' no encontrado. Saltando.")
            continue

        df = pd.read_csv(file_path, skiprows=10, encoding="latin-1")
        df = df.dropna(axis=1, how='all')
        df.columns = ["time_ms", "emg", "emg_integrated"]
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        print(f"Error: No se procesaron datos para la clase {label}.")
        return np.array([]), np.array([])

    emg = combined_df["emg"].values

    # Filtrar y rectificar la señal nativa
    emg_filtered = bandpass_filter(emg, fs)
    emg_rectified = rectify_signal(emg_filtered)

    # Ventaneo: Ventanas de 6000 ms (6.0 * 2000 Hz = 12000 muestras por ventana)
    window_size = int(6.0 * fs)
    num_windows = len(emg_rectified) // window_size

    if num_windows == 0:
        print(
            f"Advertencia: Datos insuficientes para el ventaneo en la clase {label}. Saltando."
        )
        return np.array([]), np.array([])

    # Segmentar las señales en ventanas continuas
    windows_rectified = np.array(
        [emg_rectified[i * window_size : (i + 1) * window_size] for i in range(num_windows)]
    )
    windows_filtered = np.array(
        [emg_filtered[i * window_size : (i + 1) * window_size] for i in range(num_windows)]
    )

    # Extracción de características avanzadas (Sin aplanar por envolvente lineal)
    rms = np.sqrt(np.mean(windows_filtered**2, axis=1))  # Root Mean Square
    mav = np.mean(windows_rectified, axis=1)  # Mean Absolute Value
    var = np.var(windows_filtered, axis=1)  # Varianza
    wl = np.sum(
        np.abs(np.diff(windows_filtered, axis=1)), axis=1
    )  # Waveform Length (Nueva clave)

    # Combinar en una matriz de características (Features)
    X = np.column_stack((rms, mav, var, wl))
    y = np.full(len(X), label)

    return X, y

# 4. GENERACIÓN DEL DATASET GLOBAL


X_total = []
y_total = []

for label, file_paths_for_class in files_per_class.items():
    X, y = process_files_for_class(file_paths_for_class, label)
    if X.size > 0:
        X_total.append(X)
        y_total.append(y)

if X_total and y_total:
    X_total = np.vstack(X_total)
    y_total = np.hstack(y_total)
else:
    print("No se procesaron datos para crear el dataset.")
    X_total, y_total = np.array([]), np.array([])

print("Dataset generado con éxito:")
print("-> Estructura de X (Ventanas, Características):", X_total.shape)
print("-> Estructura de y (Etiquetas):", y_total.shape)

# Mostrar distribución final de las señas
unique, counts = np.unique(y_total, return_counts=True)
print("\nDistribución de clases:")
for u, c in zip(unique, counts):
    seña = "A" if u == 0 else "B" if u == 1 else "C"
    print(f"  Clase {u} (Seña {seña}): {c} ventanas")

# Visualización de los Features extraídos
plt.figure(figsize=(10, 5))
plt.plot(X_total[:, 0], label="RMS (Energía)")
plt.plot(X_total[:, 1], label="MAV (Intensidad)")
plt.plot(X_total[:, 2], label="VAR (Varianza)")
plt.plot(X_total[:, 3], label="WL (Longitud de Onda)")
plt.title("Features del dataset optimizado (Señas LSM: A, B, C)")
plt.xlabel("Ventanas Totales")
plt.ylabel("Valor Normalizado (Visual)")
plt.legend()
plt.show()



# 5. PREPARACIÓN DE DATOS PARA LA RED NEURONAL

# Normalización estandarizada (Crucial para estabilidad de gradientes)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_total)

# Codificación One-Hot para clasificación multiclase (Softmax)
num_classes = len(np.unique(y_total))
y_categorical = to_categorical(y_total, num_classes)

# División balanceada: 85% para desarrollo entero, 15% para Test final aislado
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.15, random_state=42, stratify=y_total
)

# División interna de validación: 15% del desarrollo va a validación en vivo
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print("\nEstructura final de los conjuntos:")
print(f"-> Train: {X_train.shape}")
print(f"-> Validation: {X_val.shape}")
print(f"-> Test (Evaluación final): {X_test.shape}")


# 6. ARQUITECTURA Y CONFIGURACIÓN DE LA RED NEURONAL (ANN)

model = tf.keras.Sequential(
    [
        # Capa de Entrada ajustada a los 4 nuevos Features
        tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        # Segunda Capa Oculta de abstracción refinada
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        # Capa de Salida Multiclase (Softmax asigna probabilidades probabilísticas a A, B o C)
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compilación optimizada
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 7. ENTRENAMIENTO INTELIGENTE (CALLBACKS Y AJUSTES)=

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True  # Regresa al mejor modelo
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
)

# Ejecutar el entrenamiento con un tamaño de lote óptimo para evitar ruido excesivo
history = model.fit(
    X_train,
    y_train,
    epochs=120,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# 8. EVALUACIÓN DE RESULTADOS FINALES

# Evaluación rigurosa en el set de pruebas que la red jamás tocó
loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"ACCURACY FINAL EN DATASET DE PRUEBA: {test_accuracy*100:.2f}%")

# Gráficos de las curvas de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Pérdida (Loss)
ax1.plot(history.history["loss"], label="Train Loss", color="royalblue")
ax1.plot(history.history["val_loss"], label="Val Loss", color="orange")
ax1.set_title("Curva de Pérdida (Loss)")
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Error")
ax1.legend()
ax1.grid(True)

# Gráfico de Precisión (Accuracy)
ax2.plot(history.history["accuracy"], label="Train Acc", color="royalblue")
ax2.plot(history.history["val_accuracy"], label="Val Acc", color="orange")
ax2.set_title("Curva de Precisión (Accuracy)")
ax2.set_xlabel("Épocas")
ax2.set_ylabel("Precisión (%)")
ax2.legend()
ax2.grid(True)

plt.show()
