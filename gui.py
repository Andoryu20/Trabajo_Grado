import os
import subprocess
import threading
import time
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, ttk
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

from audio_utils import (
    verificar_ffmpeg,
    aplicar_fft,
    guardar_audio,
    obtener_nota_predominante,
    frecuencia_a_nota,
    separar_pistas
)

from visuals import (
    mostrar_espectrograma,
    mostrar_espectrogramas_separados,
    mostrar_comparacion_fft,
    
)

class AudioPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reproductor y Análisis de Audio Avanzado")
        self.audio_file = None
        self.cleaned_audio_fft = None
        self.vocal_track = None
        self.instrumental_track = None
        self.sr = 22050
        self.is_processing = False

        self.is_playing = False
        self.playback_position = 0.0
        self.current_audio = None
        self.stream = None
        self.audio_buffer = None
        self.buffer_position = 0

        self.setup_ui()
        self.setup_estilos()

    def setup_ui(self):
        self.panel_tabs = ttk.Notebook(self.root)
        self.panel_tabs.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.tab_audio = ttk.Frame(self.panel_tabs)
        self.tab_procesamiento = ttk.Frame(self.panel_tabs)
        self.tab_visualizacion = ttk.Frame(self.panel_tabs)
        self.tab_exportar = ttk.Frame(self.panel_tabs)

        self.panel_tabs.add(self.tab_audio, text="🎧 Audio")
        self.panel_tabs.add(self.tab_procesamiento, text="⚙️ Procesamiento")
        self.panel_tabs.add(self.tab_visualizacion, text="📊 Visualización")
        self.panel_tabs.add(self.tab_exportar, text="💾 Exportar")

        # AUDIO TAB
        grupo_audio = ttk.LabelFrame(self.tab_audio, text="🎧 Gestión de Audio")
        grupo_audio.pack(pady=10, padx=10, fill=tk.X)

        ttk.Button(grupo_audio, text="📂 Cargar Audio", command=self.cargar_audio).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_audio, text="▶️ Reproducir Original", command=lambda: self.iniciar_reproduccion(self.audio_file)).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_audio, text="⏯ Pausar / Reanudar", command=self.pausar_reanudar).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_audio, text="⏹ Detener", command=self.detener_audio).pack(pady=5, fill=tk.X)

        # PROCESAMIENTO TAB
        grupo_procesamiento = ttk.LabelFrame(self.tab_procesamiento, text="⚙️ Operaciones de Procesamiento")
        grupo_procesamiento.pack(pady=10, padx=10, fill=tk.X)

        ttk.Button(grupo_procesamiento, text="🛠️ Separar Pistas", command=self.separar_pistas).pack(pady=5, fill=tk.X)

        self.panel_config = ttk.LabelFrame(self.tab_procesamiento, text="⚡ Configuración FFT")
        self.panel_config.pack(pady=10, padx=10, fill=tk.X)

        self.slider_umbral = Scale(self.panel_config, from_=1, to=100, orient=tk.HORIZONTAL,
                                label="Umbral FFT (%)", length=250)
        self.slider_umbral.set(15)
        self.slider_umbral.pack(pady=10)

        # VISUALIZACIÓN TAB
        grupo_visualizacion = ttk.LabelFrame(self.tab_visualizacion, text="📊 Visualización de Resultados")
        grupo_visualizacion.pack(pady=10, padx=10, fill=tk.X)

        ttk.Button(grupo_visualizacion, text="🎼 Ver Espectrograma Original", command=self.ver_espectrograma_original).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_visualizacion, text="🎶 Ver Separación", command=self.ver_espectrogramas_separados).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_visualizacion, text="🎤 FFT Voz", command=self.ver_fft_vocal).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_visualizacion, text="🎸 FFT Instrumental", command=self.ver_fft_instrumental).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_visualizacion, text="📈 Comparativas FFT", command=self.graficar_comparativas).pack(pady=5, fill=tk.X)


        # EXPORTAR TAB
        grupo_exportar = ttk.LabelFrame(self.tab_exportar, text="💾 Exportar Audios")
        grupo_exportar.pack(pady=10, padx=10, fill=tk.X)

        ttk.Button(grupo_exportar, text="💾 Guardar Voz", command=lambda: self.guardar_audio(self.vocal_track)).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="💾 Guardar Instrumental", command=lambda: self.guardar_audio(self.instrumental_track)).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="🖼️ Guardar Espectrograma Original", command=self.guardar_espectrograma_original).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="🖼️ Guardar Separación de Pistas", command=self.guardar_espectrograma_separacion).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="🖼️ Guardar FFT Voz", command=self.guardar_fft_voz).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="🖼️ Guardar FFT Instrumental", command=self.guardar_fft_instrumental).pack(pady=5, fill=tk.X)
        ttk.Button(grupo_exportar, text="🖼️ Guardar Comparativas FFT", command=self.guardar_comparativas_fft).pack(pady=5, fill=tk.X)


        # Barra de estado
        self.barra_estado = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)


    def setup_estilos(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')

        primary_color = '#00aae4'       # Morado intenso (color principal)
        accent_color = '#03dac6'         # Verde agua (acento)
        background_color = '#f5f5f5'     # Fondo general claro
        tab_active_color = '#bb86fc'     # Color tab activo
        button_active_color = '#90caf9'  # Botón al presionar
        text_color = '#000000'           # Texto negro

        # Estilos para Tabs
        style.configure('TNotebook', background=background_color, borderwidth=0)
        style.configure('TNotebook.Tab', background=background_color, foreground=text_color, padding=[12, 6], font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', tab_active_color)], foreground=[('selected', text_color)])

        # Estilos para Frames y Labels
        style.configure('TFrame', background=background_color)
        style.configure('TLabelFrame', background=background_color, foreground=text_color, font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background=background_color, foreground=text_color)

        # Estilos para Botones
        style.configure('TButton',
                        font=('Segoe UI', 10, 'bold'),
                        padding=8,
                        background=primary_color,
                        foreground='white',
                        borderwidth=0,
                        relief="flat")
        style.map('TButton',
                background=[('active', button_active_color)],
                foreground=[('active', 'white')])

        # Barra de estado
        self.barra_estado.config(background=primary_color, foreground='white', font=('Segoe UI', 9, 'bold'))


    def cargar_audio(self):
        rutas_audio = [("Archivos de audio", "*.wav *.mp3 *.flac *.ogg")]
        ruta_archivo = filedialog.askopenfilename(filetypes=rutas_audio)

        if ruta_archivo:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if not ruta_archivo.endswith(".wav"):
                        audio = AudioSegment.from_file(ruta_archivo)
                        ruta_archivo = "temp_conversion.wav"
                        audio.export(ruta_archivo, format="wav")

                    self.audio_file, self.sr = librosa.load(ruta_archivo, sr=self.sr, res_type='kaiser_fast')
                    self.actualizar_estado(f"Audio cargado: {os.path.basename(ruta_archivo)}")

                    if os.path.exists("temp_conversion.wav"):
                        os.remove("temp_conversion.wav")

            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar audio:\n{str(e)}")

    def separar_pistas(self):
        if self.audio_file is None:
            messagebox.showwarning("Advertencia", "Primero carga un archivo de audio")
            return
        try:
            self.is_processing = True
            self.actualizar_estado("Iniciando separación de pistas...")
            
            ventana = self.mostrar_cargando("Separando pistas...")

            self.vocal_track, self.instrumental_track = separar_pistas(self.audio_file, self.sr)

            ventana.destroy()

            self.actualizar_estado("Separación completada exitosamente")
            messagebox.showinfo("Éxito", "Pistas separadas correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"Error en separación:\n{str(e)}")
        finally:
            self.is_processing = False

    def guardar_audio(self, pista):
        if pista is None:
            messagebox.showwarning("Advertencia", "No hay pista para guardar")
            return

        opciones = [("Archivo WAV", "*.wav"), ("Todos los archivos", "*.*")]
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=opciones)

        if ruta_guardado:
            try:
                guardar_audio(pista, self.sr, ruta_guardado)
                self.actualizar_estado(f"Archivo guardado: {os.path.basename(ruta_guardado)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar:\n{str(e)}")

    def ver_espectrograma_original(self):
        if self.audio_file is not None:
            mostrar_espectrograma(self.audio_file, self.sr, "Original")
        else:
            messagebox.showwarning("Advertencia", "Primero carga un audio")

    def ver_espectrogramas_separados(self):
        if self.vocal_track is not None and self.instrumental_track is not None:
            mostrar_espectrogramas_separados(self.vocal_track, self.instrumental_track, self.sr)
        else:
            messagebox.showwarning("Advertencia", "Primero separa las pistas")

    def ver_fft_vocal(self):
        if self.vocal_track is None:
            messagebox.showwarning("Advertencia", "Primero separa las pistas")
            return

        # Verificar si ya se aplicó FFT
        if np.array_equal(self.vocal_track, aplicar_fft(self.vocal_track.copy(), 0)):
            messagebox.showwarning("Advertencia", "Debes aplicar la FFT a la pista vocal antes de visualizarla")
            return

        original = self.vocal_track.copy()
        procesado = aplicar_fft(original, self.slider_umbral.get())
        mostrar_comparacion_fft(original, procesado, self.sr, "Voz")


    def ver_fft_instrumental(self):
        if self.instrumental_track is None:
            messagebox.showwarning("Advertencia", "Primero separa las pistas")
            return

        if np.array_equal(self.instrumental_track, aplicar_fft(self.instrumental_track.copy(), 0)):
            messagebox.showwarning("Advertencia", "Debes aplicar la FFT a la pista instrumental antes de visualizarla")
            return

        original = self.instrumental_track.copy()
        procesado = aplicar_fft(original, self.slider_umbral.get())
        mostrar_comparacion_fft(original, procesado, self.sr, "Instrumental")


    def graficar_comparativas(self):
        try:
            if self.vocal_track is None or self.instrumental_track is None:
                messagebox.showwarning("Advertencia", "Primero separa las pistas")
                return

            # Aplicar FFT en tiempo real según el umbral seleccionado
            vocal_fft = aplicar_fft(self.vocal_track.copy(), self.slider_umbral.get())
            instrumental_fft = aplicar_fft(self.instrumental_track.copy(), self.slider_umbral.get())

            plt.figure(figsize=(14, 8))

            # --- Graficar espectrograma de la voz ---
            plt.subplot(2, 1, 1)
            D_vocal = librosa.amplitude_to_db(np.abs(librosa.stft(vocal_fft)), ref=np.max)
            librosa.display.specshow(D_vocal, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
            plt.colorbar(format='%+2.0f dB')
            
            nota_vocal = obtener_nota_predominante(vocal_fft, self.sr)
            plt.title(f"Voz - FFT Aplicada - Nota Predominante: {nota_vocal}", fontsize=14, fontweight='bold')
            plt.xlabel("")
            plt.ylabel("Frecuencia (Hz)")

            # --- Graficar espectrograma del instrumental ---
            plt.subplot(2, 1, 2)
            D_instrumental = librosa.amplitude_to_db(np.abs(librosa.stft(instrumental_fft)), ref=np.max)
            librosa.display.specshow(D_instrumental, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
            plt.colorbar(format='%+2.0f dB')

            nota_instrumental = obtener_nota_predominante(instrumental_fft, self.sr)
            plt.title(f"Instrumental - FFT Aplicada - Nota Predominante: {nota_instrumental}", fontsize=14, fontweight='bold')
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Frecuencia (Hz)")

            plt.tight_layout(pad=2.0)
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar:\n{str(e)}")

    def guardar_espectrograma_original(self):
        try:
            if self.audio_file is None:
                messagebox.showwarning("Advertencia", "Primero carga un audio")
                return

            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if ruta_guardado:
                plt.figure(figsize=(12, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_file)), ref=np.max)
                librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Espectrograma Original")
                plt.tight_layout()
                plt.savefig(ruta_guardado)
                plt.close()
                messagebox.showinfo("Éxito", f"Gráfico guardado exitosamente:\n{ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el espectrograma:\n{str(e)}")

    def guardar_espectrograma_separacion(self):
        try:
            if self.vocal_track is None or self.instrumental_track is None:
                messagebox.showwarning("Advertencia", "Primero separa las pistas")
                return

            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if ruta_guardado:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                D_vocal = librosa.amplitude_to_db(np.abs(librosa.stft(self.vocal_track)), ref=np.max)
                librosa.display.specshow(D_vocal, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Espectrograma Voz Separada")

                plt.subplot(2, 1, 2)
                D_instrumental = librosa.amplitude_to_db(np.abs(librosa.stft(self.instrumental_track)), ref=np.max)
                librosa.display.specshow(D_instrumental, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Espectrograma Instrumental Separado")

                plt.tight_layout()
                plt.savefig(ruta_guardado)
                plt.close()
                messagebox.showinfo("Éxito", f"Gráfico guardado exitosamente:\n{ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el espectrograma de separación:\n{str(e)}")

    def guardar_fft_voz(self):
        try:
            if self.vocal_track is None:
                messagebox.showwarning("Advertencia", "Primero separa las pistas")
                return

            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if ruta_guardado:
                plt.figure(figsize=(12, 4))
                original = self.vocal_track.copy()
                procesado = aplicar_fft(original, self.slider_umbral.get())

                D = librosa.amplitude_to_db(np.abs(librosa.stft(procesado)), ref=np.max)
                librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("FFT Aplicada a Voz")
                plt.tight_layout()
                plt.savefig(ruta_guardado)
                plt.close()
                messagebox.showinfo("Éxito", f"Gráfico guardado exitosamente:\n{ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la FFT de voz:\n{str(e)}")

    def guardar_fft_instrumental(self):
        try:
            if self.instrumental_track is None:
                messagebox.showwarning("Advertencia", "Primero separa las pistas")
                return

            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if ruta_guardado:
                plt.figure(figsize=(12, 4))
                original = self.instrumental_track.copy()
                procesado = aplicar_fft(original, self.slider_umbral.get())

                D = librosa.amplitude_to_db(np.abs(librosa.stft(procesado)), ref=np.max)
                librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("FFT Aplicada a Instrumental")
                plt.tight_layout()
                plt.savefig(ruta_guardado)
                plt.close()
                messagebox.showinfo("Éxito", f"Gráfico guardado exitosamente:\n{ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la FFT de instrumental:\n{str(e)}")

    def guardar_comparativas_fft(self):
        try:
            if self.vocal_track is None or self.instrumental_track is None:
                messagebox.showwarning("Advertencia", "Primero separa las pistas")
                return

            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if ruta_guardado:
                vocal_fft = aplicar_fft(self.vocal_track.copy(), self.slider_umbral.get())
                instrumental_fft = aplicar_fft(self.instrumental_track.copy(), self.slider_umbral.get())

                plt.figure(figsize=(14, 8))

                plt.subplot(2, 1, 1)
                D_vocal = librosa.amplitude_to_db(np.abs(librosa.stft(vocal_fft)), ref=np.max)
                librosa.display.specshow(D_vocal, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Voz - FFT Aplicada")

                plt.subplot(2, 1, 2)
                D_instrumental = librosa.amplitude_to_db(np.abs(librosa.stft(instrumental_fft)), ref=np.max)
                librosa.display.specshow(D_instrumental, sr=self.sr, x_axis='time', y_axis='log', cmap='plasma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Instrumental - FFT Aplicada")

                plt.tight_layout()
                plt.savefig(ruta_guardado)
                plt.close()
                messagebox.showinfo("Éxito", f"Gráfico guardado exitosamente:\n{ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar las comparativas FFT:\n{str(e)}")


    def iniciar_reproduccion(self, audio):
        if audio is None:
            messagebox.showwarning("Advertencia", "No hay audio para reproducir")
            return

        if self.current_audio is not audio:
            self.playback_position = 0.0
            self.current_audio = audio
            self.audio_buffer = audio.copy()
            self.buffer_position = 0

        if self.is_playing:
            self.pausar_reanudar()
            return

        self.reproducir_audio()

    def reproducir_audio(self):
        try:
            if self.stream is not None:
                self.stream.close()

            start_sample = int(self.playback_position * self.sr)
            self.audio_buffer = self.current_audio[start_sample:]
            self.buffer_position = 0

            def callback(outdata, frames, time_info, status):
                if self.is_playing:
                    remaining = len(self.audio_buffer) - self.buffer_position
                    if remaining <= 0:
                        outdata[:] = 0
                        raise sd.CallbackStop

                    chunksize = min(remaining, frames)
                    outdata[:chunksize] = self.audio_buffer[
                        self.buffer_position:self.buffer_position + chunksize
                    ].reshape(-1, 1)
                    outdata[chunksize:] = 0
                    self.buffer_position += chunksize
                    self.playback_position += chunksize / self.sr
                else:
                    raise sd.CallbackStop

            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                callback=callback,
                finished_callback=self.finalizar_reproduccion
            )
            self.stream.start()
            self.is_playing = True
            self.actualizar_estado("Reproduciendo...")

        except Exception as e:
            self.actualizar_estado(f"Error en reproducción: {str(e)}")
            self.is_playing = False

    def pausar_reanudar(self):
        if self.is_playing:
            self.is_playing = False
            if self.stream is not None:
                self.stream.stop()
            self.actualizar_estado("Reproducción pausada")
        else:
            self.is_playing = True
            if self.stream is not None:
                self.stream.start()
            self.actualizar_estado("Reanudando reproducción...")

    def detener_audio(self):
        self.is_playing = False
        self.playback_position = 0.0
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.actualizar_estado("Reproducción detenida")

    def finalizar_reproduccion(self):
        self.is_playing = False
        self.playback_position = 0.0
        self.actualizar_estado("Reproducción finalizada")

    def actualizar_estado(self, mensaje):
        self.barra_estado.config(text=mensaje)
        self.root.update_idletasks()

    def mostrar_cargando(self, mensaje):
        ventana = tk.Toplevel(self.root)
        ventana.title("Procesando...")
        ventana.geometry("250x80")
        ventana.resizable(False, False)
        tk.Label(ventana, text=mensaje, font=("Segoe UI", 11)).pack(pady=20)
        ventana.update()
        return ventana

AudioPlayerApp = AudioPlayerApp

