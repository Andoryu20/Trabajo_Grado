import os
import subprocess
import warnings
import numpy as np
import librosa
import soundfile as sf
from scipy.fft import fft, ifft
from pydub import AudioSegment

def verificar_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise EnvironmentError("FFmpeg no encontrado. Descárgalo en: https://ffmpeg.org/download.html")

def aplicar_fft(audio, umbral_porcentaje):
    if audio is None:
        return None
    
    fft_transform = fft(audio)
    umbral = (umbral_porcentaje / 100) * np.max(np.abs(fft_transform))
    fft_transform[np.abs(fft_transform) < umbral] = 0
    return np.real(ifft(fft_transform))

def normalizar_audio(audio):
    if audio is None or np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))

def guardar_audio(pista, sr, ruta_guardado):
    if pista is None:
        raise ValueError("No hay pista para guardar")
    pista_normalizada = normalizar_audio(pista)
    sf.write(ruta_guardado, pista_normalizada, sr)

def obtener_nota_predominante(audio, sr):
    try:
        f0 = librosa.yin(audio, fmin=80, fmax=1000, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return "No detectada"
        return frecuencia_a_nota(np.median(f0))
    except:
        return "Error en detección"

def frecuencia_a_nota(frecuencia):
    if frecuencia <= 0:
        return "N/A"
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    num_midi = 12 * np.log2(frecuencia / 440.0) + 69
    octava = int((num_midi // 12) - 1)
    return f"{notas[int(num_midi % 12)]}{octava}"

def separar_pistas(audio_array, sr, temp_name="temp_audio"):
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{temp_name}.wav")
    sf.write(temp_path, audio_array, sr)

    comando = [
        "demucs",
        "--two-stems", "vocals",
        "-o", "separated",
        temp_path
    ]
    proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proceso.communicate()

    base_path = os.path.join("separated", "htdemucs", temp_name)
    vocal_path = os.path.join(base_path, "vocals.wav")
    instrumental_path = os.path.join(base_path, "other.wav")

    if not os.path.exists(vocal_path) or not os.path.exists(instrumental_path):
        raise FileNotFoundError("No se encontraron las pistas separadas")

    vocal_track, _ = librosa.load(vocal_path, sr=sr)
    instrumental_track, _ = librosa.load(instrumental_path, sr=sr)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return vocal_track, instrumental_track

