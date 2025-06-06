# visuals.py

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from audio_utils import obtener_nota_predominante

# 1. Primero graficar_espectrograma
def graficar_espectrograma(audio, sr, titulo):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='plasma')
    plt.colorbar(format='%+2.0f dB')
    nota = obtener_nota_predominante(audio, sr)
    plt.title(f"{titulo} - Nota Predominante: {nota}", fontsize=12, fontweight="bold")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")


# 2. Luego todas las funciones que la usan
def mostrar_espectrograma(audio, sr, titulo):
    plt.figure(figsize=(12, 4))
    graficar_espectrograma(audio, sr, titulo)
    plt.tight_layout()
    plt.show()

def mostrar_espectrogramas_separados(vocal_track, instrumental_track, sr):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    graficar_espectrograma(vocal_track, sr, "Voz - Original")

    plt.subplot(2, 1, 2)
    graficar_espectrograma(instrumental_track, sr, "Instrumental - Original")

    plt.tight_layout()
    plt.show()


def mostrar_comparacion_fft(original, procesado, sr, titulo):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    graficar_espectrograma(original, sr, f"{titulo} Original")
    plt.subplot(2, 1, 2)
    graficar_espectrograma(procesado, sr, f"{titulo} con FFT")
    plt.tight_layout()
    plt.show()
