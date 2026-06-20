"""
voice.py — Síntesis de voz con entrenamiento personalizado usando matemática pura.

Objetivo: tomar un video/audio de referencia, aprender una voz, y sintetizar
nuevas frases con la mayor naturalidad posible SIN depender de librerías TTS
pre-entrenadas de alto nivel (MeloTTS, Coqui, etc.).

Componentes matemáticos implementados desde cero:
  • Preprocesamiento de texto -> caracteres/fonemas.
  • Extracción de audio con yt-dlp + ffmpeg.
  • STFT, banco de filtros Mel, log-Mel espectrograma (PyTorch).
  • Estimación de F0 por autocorrelación (YIN simplificado) y energía.
  • Alineación texto-audio mediante Dynamic Time Warping (DTW).
  • Modelo acústico tipo FastSpeech 2 (transformer/FFT blocks, predictores de
    duración, pitch y energía, Length Regulator).
  • Vocoder inverso: inversión de Mel + Griffin-Lim.
  • Post-procesamiento: preénfasis/inverso, normalización, escritura WAV.

Requiere: numpy, torch, scipy (solo para DTW; se puede reemplazar por una
implementación propia si se desea).
"""

import json
import math
import os
import struct
import subprocess
import tempfile
import time
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# scipy es opcional; usado para DTW. Si no está disponible, se implementa naive.
try:
    from scipy.spatial.distance import cdist
    from scipy.signal import lfilter
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuración global y dispositivo
# -----------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.backends.cuda.is_available() else "cpu"))
print(f"[*] Dispositivo de aceleración: {DEVICE}")

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

MAX_WAV_VALUE = 32768.0
PREEMPHASIS = 0.97


# -----------------------------------------------------------------------------
# Utilidades de audio
# -----------------------------------------------------------------------------
def _which(cmd: str) -> bool:
    return subprocess.run(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def load_audio(source: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Carga audio desde una URL de video (yt-dlp) o desde una ruta local WAV.
    Devuelve onda mono PCM float32 normalizada.
    Requiere ffmpeg; yt-dlp solo si source es una URL.
    """
    source = source.strip()
    es_url = source.startswith(("http://", "https://"))

    if es_url and not _which("yt-dlp"):
        raise RuntimeError("Se requiere 'yt-dlp' en el PATH para descargar URLs.")
    if not _which("ffmpeg"):
        raise RuntimeError("Se requiere 'ffmpeg' en el PATH.")

    tmp_path = tempfile.mktemp(suffix=".raw")
    try:
        if es_url:
            ytdlp_cmd = ["yt-dlp", "-f", "bestaudio", "--audio-quality", "0", "-o", "-", source]
            ffmpeg_input = ["-i", "pipe:0"]
            p1 = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            stdin = p1.stdout
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"No se encontró el archivo de audio: {source}")
            ffmpeg_input = ["-i", source]
            p1 = None
            stdin = None

        ffmpeg_cmd = [
            "ffmpeg", "-y", *ffmpeg_input,
            "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", tmp_path
        ]
        p2 = subprocess.Popen(ffmpeg_cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p1 is not None:
            p1.stdout.close()
        p2.communicate()

        with open(tmp_path, "rb") as f:
            raw = f.read()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / MAX_WAV_VALUE
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        return audio
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_audio(url: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compatibilidad hacia atrás: descarga audio desde URL."""
    return load_audio(url, sr=sr)


def preemphasis(wave: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    """Filtro de preénfasis para compensar la caída de alta frecuencia."""
    if HAS_SCIPY:
        return lfilter([1.0, -coeff], [1.0], wave)
    out = np.zeros_like(wave)
    out[0] = wave[0]
    out[1:] = wave[1:] - coeff * wave[:-1]
    return out


def inv_preemphasis(wave: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    """Inverso del preénfasis."""
    if HAS_SCIPY:
        return lfilter([1.0], [1.0, -coeff], wave)
    out = np.zeros_like(wave)
    out[0] = wave[0]
    for i in range(1, len(wave)):
        out[i] = wave[i] + coeff * out[i - 1]
    return out


def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Escribe audio normalizado a WAV PCM 16-bit con cabecera RIFF."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    data = pcm.tobytes()
    byte_rate = sr * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ",
        16, 1, 1, sr, byte_rate, 2, 16, b"data", len(data)
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)


# -----------------------------------------------------------------------------
# Procesamiento de texto
# -----------------------------------------------------------------------------
# Alfabeto simplificado para español (carácter -> índice). No usa librerías TTS.
ALPHABET = (
    "_" +  # padding
    " " +
    "abcdefghijklmnopqrstuvwxyz" +
    "ñ" +
    "áéíóúü" +
    ".,!?-"
)
PAD_IDX = 0


def text_to_sequence(text: str) -> List[int]:
    """Convierte texto a secuencia de enteros según el alfabeto."""
    seq = []
    for ch in text.lower():
        if ch in ALPHABET:
            seq.append(ALPHABET.index(ch))
        elif ch.isdigit():
            # Para números, los deletreamos como palabras simplificadas.
            seq.extend(text_to_sequence(_spell_digit(ch)))
    if not seq:
        seq = [ALPHABET.index(" ")]
    return seq


def _spell_digit(d: str) -> str:
    mapping = {
        "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
        "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve"
    }
    return mapping.get(d, "")


def sequence_to_text(seq: List[int]) -> str:
    return "".join(ALPHABET[idx] for idx in seq if 0 <= idx < len(ALPHABET))


# -----------------------------------------------------------------------------
# DSP: mel-espectrograma, F0 y energía
# -----------------------------------------------------------------------------
def _mel_scale(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _inv_mel_scale(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filter_bank(sr: int, n_fft: int, n_mels: int,
                          f_min: float, f_max: float) -> np.ndarray:
    """Construye el banco de filtros Mel con la fórmula clásica."""
    low_mel = _mel_scale(f_min)
    high_mel = _mel_scale(f_max)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _inv_mel_scale(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        # Rama ascendente
        if center > left:
            k = np.arange(left, center)
            filters[m - 1, left:center] = (k - left) / (center - left)
        # Rama descendente
        if right > center:
            k = np.arange(center, right)
            filters[m - 1, center:right] = (right - k) / (right - center)
    return filters


class DSP:
    """Capa de procesamiento digital de señales basada en PyTorch."""

    def __init__(self,
                 sr: int = SAMPLE_RATE,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH,
                 n_mels: int = N_MELS,
                 f_min: float = MEL_FMIN,
                 f_max: float = MEL_FMAX):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        mel_filters = build_mel_filter_bank(sr, n_fft, n_mels, f_min, f_max)
        self.register_buffers(mel_filters)

    def register_buffers(self, mel_filters: np.ndarray):
        self.mel_filters_t = torch.from_numpy(mel_filters).float().to(DEVICE)
        self.window_t = torch.hann_window(self.win_length).to(DEVICE)

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convierte una onda [samples] o [batch, samples] en log-Mel [batch, n_mels, frames].
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # Padding simétrico para centrar ventanas
        pad_amount = (self.n_fft - self.hop_length) // 2
        wav = F.pad(wav, (pad_amount, pad_amount), mode="reflect")

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_t,
            return_complex=True,
            center=False,
        )
        mag = torch.abs(stft)
        power = mag ** 2.0
        mel = torch.matmul(self.mel_filters_t, power)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    def wav_to_mel_numpy(self, wav: np.ndarray) -> np.ndarray:
        wav_t = torch.from_numpy(wav).float().to(DEVICE)
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)
        with torch.no_grad():
            mel = self.wav_to_mel(wav_t)
        return mel.squeeze(0).cpu().numpy()

    def energy(self, wav: np.ndarray) -> np.ndarray:
        """Energía por frame calculada sobre la onda enventanada."""
        wav = np.copy(wav)
        n_frames = 1 + (len(wav) - self.win_length) // self.hop_length
        energies = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            frame = wav[i * self.hop_length: i * self.hop_length + self.win_length]
            frame *= np.hanning(self.win_length)
            energies[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
        return energies

    def pitch(self, wav: np.ndarray) -> np.ndarray:
        """
        Estimación de F0 mediante autocorrelación por YIN simplificado.
        Devuelve F0 por frame en Hz (0 para no voicing).
        """
        sr = self.sr
        hop = self.hop_length
        win = self.win_length
        min_period = max(1, int(sr / 500.0))   # ~500 Hz max
        max_period = min(win // 2, int(sr / 50.0))  # ~50 Hz min

        n_frames = 1 + (len(wav) - win) // hop
        f0 = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            frame = wav[i * hop: i * hop + win]
            frame = frame - np.mean(frame)
            frame = frame * np.hanning(win)

            # Diferencia cuadrática (YIN) para robustez
            diff = np.zeros(max_period, dtype=np.float32)
            for tau in range(1, max_period):
                diff[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2)

            # Función de autocorrelación normalizada (CMNDF)
            cmndf = np.zeros(max_period, dtype=np.float32)
            cum = 0.0
            for tau in range(1, max_period):
                cum += diff[tau]
                cmndf[tau] = diff[tau] / ((cum / tau) + 1e-12)

            # Buscar primer mínimo por debajo del umbral
            threshold = 0.1
            best_tau = 0
            for tau in range(min_period, max_period - 1):
                if cmndf[tau] < threshold and cmndf[tau] < cmndf[tau - 1] and cmndf[tau] < cmndf[tau + 1]:
                    best_tau = tau
                    break

            if best_tau > 0:
                # Parábola cuadrática para interpolar y refinar tau
                y1, y2, y3 = cmndf[best_tau - 1], cmndf[best_tau], cmndf[best_tau + 1]
                denom = 2.0 * (y1 - 2.0 * y2 + y3)
                if abs(denom) > 1e-12:
                    delta = (y1 - y3) / denom
                    best_tau = best_tau + delta
                f0[i] = sr / best_tau
        return f0


# -----------------------------------------------------------------------------
# Alineación texto-audio con DTW
# -----------------------------------------------------------------------------
def _dtw(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    DTW clásico entre dos secuencias (frames x features).
    Devuelve el path (índices de y para cada x) y la distancia acumulada.
    """
    if HAS_SCIPY:
        # Distancia euclídea por frame
        cost = cdist(x, y, metric="euclidean")
    else:
        cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    # Backtracking
    path_y = np.zeros(n, dtype=int)
    i, j = n, m
    while i > 0 and j > 0:
        path_y[i - 1] = j - 1
        choices = {
            (i - 1, j - 1): acc[i - 1, j - 1],
            (i - 1, j): acc[i - 1, j],
            (i, j - 1): acc[i, j - 1],
        }
        ni, nj = min(choices, key=choices.get)
        if ni == i:
            j = nj
        elif nj == j:
            i = ni
        else:
            i, j = ni, nj

    return path_y, acc[n, m]


def align_text_to_mel(chars: List[int], mel: np.ndarray) -> List[int]:
    """
    Alinea cada carácter de texto con una cantidad de frames de mel.
    Retorna una lista de duraciones (en frames) por carácter.
    """
    n_chars = len(chars)
    n_frames = mel.shape[1]
    if n_chars == 0 or n_frames == 0:
        return [1] * max(n_chars, 1)

    # Crear embeddings simples de carácter: one-hot-ish posicional
    char_feats = np.zeros((n_chars, 8), dtype=np.float32)
    for i, c in enumerate(chars):
        char_feats[i, 0] = c / len(ALPHABET)
        char_feats[i, 1] = i / max(n_chars, 1)
        char_feats[i, 2] = np.sin(i * 0.1)
        char_feats[i, 3] = np.cos(i * 0.1)

    # Reducir mel a la misma dimensión de características que char_feats (8)
    # para que DTW pueda comparar ambas secuencias.
    mel_T = mel.T.astype(np.float32)
    n_mels = mel_T.shape[1]
    # Dividimos los 80 bins mel en 8 grupos y promediamos.
    groups = 8
    bins_per_group = n_mels // groups
    mel_reduced = np.zeros((n_frames, groups), dtype=np.float32)
    for g in range(groups):
        start = g * bins_per_group
        end = start + bins_per_group if g < groups - 1 else n_mels
        mel_reduced[:, g] = mel_T[:, start:end].mean(axis=1)

    mel_mean = mel_reduced.mean(axis=0, keepdims=True)
    mel_std = mel_reduced.std(axis=0, keepdims=True) + 1e-8
    mel_norm = (mel_reduced - mel_mean) / mel_std

    path, _ = _dtw(char_feats, mel_norm)

    # Contar cuántos frames le toca a cada carácter
    durations = [0] * n_chars
    for frame_idx in path:
        if 0 <= frame_idx < n_chars:
            durations[frame_idx] += 1

    # Asegurar que cada carácter tenga al menos 1 frame
    durations = [max(1, d) for d in durations]
    return durations


# -----------------------------------------------------------------------------
# Modelo acústico: arquitectura tipo FastSpeech 2
# -----------------------------------------------------------------------------
def positional_encoding(length: int, channels: int, device: torch.device) -> torch.Tensor:
    """Codificación posicional sinusoidal."""
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32, device=device) *
        (-math.log(10000.0) / channels)
    )
    pe = torch.zeros(length, channels, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class FFTBlock(nn.Module):
    """Bloque Feed-Forward Transformer: conv1d + ReLU + conv1d + layer norm + dropout."""

    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels * 4, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels * 4, channels, kernel_size, padding=padding)
        self.ln = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, length, channels]
        residual = x
        x = x.transpose(1, 2)  # [batch, channels, length]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [batch, length, channels]
        x = self.ln(x + residual)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class LengthRegulator(nn.Module):
    """Expande los estados de carácter según las duraciones predichas."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        # x: [batch, char_len, channels]
        # durations: [batch, char_len] (enteros o flotantes redondeados)
        output = []
        for b in range(x.size(0)):
            expanded = []
            for i in range(x.size(1)):
                repeat = int(torch.round(durations[b, i]).item())
                repeat = max(1, repeat)
                expanded.append(x[b, i:i + 1].repeat(repeat, 1))
            expanded = torch.cat(expanded, dim=0)
            output.append(expanded)

        if max_len is not None:
            out_lens = [o.size(0) for o in output]
            max_len = max(max(out_lens), max_len)
        else:
            max_len = max(o.size(0) for o in output)

        padded = []
        for o in output:
            if o.size(0) < max_len:
                pad = torch.zeros(max_len - o.size(0), o.size(1), device=o.device, dtype=o.dtype)
                o = torch.cat([o, pad], dim=0)
            else:
                o = o[:max_len]
            padded.append(o)
        return torch.stack(padded, dim=0)


class VariancePredictor(nn.Module):
    """Predice duración, pitch o energía: conv + layer norm + dropout + proyección."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.ln2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, length, channels]
        residual = x
        h = x.transpose(1, 2)
        h = self.conv1(h)
        h = h.transpose(1, 2)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.ln2(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = h + residual
        out = self.proj(h).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class AcousticModel(nn.Module):
    """
    Modelo TTS basado en FastSpeech 2, entrenado desde cero con matemática.
    Predice espectrograma mel, pitch y energía a partir de texto.
    """

    def __init__(self,
                 vocab_size: int = len(ALPHABET),
                 channels: int = 256,
                 n_fft_blocks: int = 4,
                 mel_dim: int = N_MELS,
                 dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.mel_dim = mel_dim

        self.embedding = nn.Embedding(vocab_size, channels, padding_idx=PAD_IDX)

        # Encoder de caracteres
        self.encoder_layers = nn.ModuleList([
            FFTBlock(channels, dropout=dropout) for _ in range(n_fft_blocks)
        ])

        # Predictores de varianza
        self.duration_predictor = VariancePredictor(channels)
        self.pitch_predictor = VariancePredictor(channels)
        self.energy_predictor = VariancePredictor(channels)

        # Embeddings condicionales de pitch/energy
        self.pitch_bins = nn.Parameter(torch.linspace(math.log(50.0), math.log(600.0), 256).exp(), requires_grad=False)
        self.pitch_embedding = nn.Embedding(256, channels)
        self.energy_bins = nn.Parameter(torch.linspace(0.0, 1.0, 256), requires_grad=False)
        self.energy_embedding = nn.Embedding(256, channels)

        # Regulador de longitud
        self.length_regulator = LengthRegulator()

        # Decoder de mel (más FFT blocks)
        self.decoder_layers = nn.ModuleList([
            FFTBlock(channels, dropout=dropout) for _ in range(n_fft_blocks)
        ])

        self.mel_projection = nn.Linear(channels, mel_dim)

    def _quantize(self, values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Cuantiza valores continuos a índices de bins."""
        # values: [batch, len], bins: [n_bins]
        distances = torch.abs(values.unsqueeze(-1) - bins.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1)
        return indices

    def forward(self,
                tokens: torch.Tensor,
                durations: Optional[torch.Tensor] = None,
                target_mel_len: Optional[int] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None):
        # tokens: [batch, char_len]
        x = self.embedding(tokens) * math.sqrt(self.channels)
        bsz, char_len, _ = x.shape
        pe = positional_encoding(char_len, self.channels, x.device)
        x = x + pe.unsqueeze(0)

        mask = tokens == PAD_IDX

        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        # Predicciones en nivel de carácter
        log_duration_pred = self.duration_predictor(x, mask=mask)
        pitch_char_pred = self.pitch_predictor(x, mask=mask)
        energy_char_pred = self.energy_predictor(x, mask=mask)

        # Duraciones: entrenamos con log(dur) para estabilidad
        if durations is not None:
            durations_for_reg = durations.float()
        else:
            durations_for_reg = torch.exp(log_duration_pred) - 1.0
            durations_for_reg = torch.clamp(durations_for_reg, min=1.0)

        # Expandir a nivel de frame
        x_expanded = self.length_regulator(x, durations_for_reg, max_len=target_mel_len)
        mel_len = x_expanded.size(1)

        # Expander pitch/energy al nivel de frame.
        # Si los targets vienen dados (entrenamiento), ya están a nivel de frame
        # (longitud de mel), así que solo los recortamos/empadronamos.
        # En inferencia usamos las predicciones por carácter y las expandemos.
        if pitch_target is not None and energy_target is not None:
            pitch_frame = self._pad_or_truncate(pitch_target, mel_len)
            energy_frame = self._pad_or_truncate(energy_target, mel_len)
        else:
            pitch_frame = self._expand_variance(pitch_char_pred, durations_for_reg, mel_len)
            energy_frame = self._expand_variance(energy_char_pred, durations_for_reg, mel_len)

        # Embeddings condicionales
        pitch_idx = self._quantize(pitch_frame, self.pitch_bins)
        energy_idx = self._quantize(energy_frame, self.energy_bins)
        pitch_emb = self.pitch_embedding(pitch_idx)
        energy_emb = self.energy_embedding(energy_idx)

        x_expanded = x_expanded + pitch_emb + energy_emb
        x_expanded = x_expanded + positional_encoding(mel_len, self.channels, x_expanded.device).unsqueeze(0)

        for layer in self.decoder_layers:
            x_expanded = layer(x_expanded)

        mel_pred = self.mel_projection(x_expanded)
        return mel_pred, log_duration_pred, pitch_char_pred, energy_char_pred

    @staticmethod
    def _expand_variance(var_per_char: torch.Tensor,
                         durations: torch.Tensor,
                         max_len: int) -> torch.Tensor:
        """Repite el valor de varianza por carácter según duración."""
        out = []
        for b in range(var_per_char.size(0)):
            expanded = []
            for i in range(var_per_char.size(1)):
                repeat = int(torch.round(durations[b, i]).item())
                repeat = max(1, repeat)
                expanded.append(var_per_char[b, i].repeat(repeat))
            expanded = torch.cat(expanded, dim=0)
            if expanded.size(0) < max_len:
                pad = torch.zeros(max_len - expanded.size(0), device=expanded.device, dtype=expanded.dtype)
                expanded = torch.cat([expanded, pad], dim=0)
            else:
                expanded = expanded[:max_len]
            out.append(expanded)
        return torch.stack(out, dim=0)

    @staticmethod
    def _pad_or_truncate(var_per_frame: torch.Tensor, max_len: int) -> torch.Tensor:
        """Recorta o empadrona un tensor [batch, frame_len] a [batch, max_len]."""
        batch, frame_len = var_per_frame.shape
        if frame_len < max_len:
            pad = torch.zeros(batch, max_len - frame_len, device=var_per_frame.device, dtype=var_per_frame.dtype)
            return torch.cat([var_per_frame, pad], dim=1)
        return var_per_frame[:, :max_len]


# -----------------------------------------------------------------------------
# Vocoder clásico: inversión de Mel + Griffin-Lim (fallback)
# -----------------------------------------------------------------------------
def mel_to_linear(mel: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """
    mel: [n_mels, frames] log-mel (log natural o log10).
    Devuelve espectrograma de magnitud lineal [n_fft//2+1, frames].
    """
    mel_linear = np.exp(mel)  # asume log natural usado en entrenamiento
    # Pseudo-inversa del banco mel
    filters_pinv = np.linalg.pinv(mel_filters)
    mag = np.maximum(np.dot(filters_pinv, mel_linear), 1e-5)
    return mag


def griffin_lim(mag: np.ndarray,
                n_fft: int = N_FFT,
                hop_length: int = HOP_LENGTH,
                win_length: int = WIN_LENGTH,
                n_iter: int = 60) -> np.ndarray:
    """
    Griffin-Lim para reconstruir fase a partir de magnitud.
    """
    n_frames = mag.shape[1]
    expected_len = n_frames * hop_length + win_length
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    stft_matrix = mag * angles
    window = np.hanning(win_length)

    for _ in range(n_iter):
        audio = np.zeros(expected_len)
        for i in range(n_frames):
            start = i * hop_length
            audio[start:start + win_length] += np.fft.irfft(stft_matrix[:, i], n=n_fft)[:win_length] * window

        # Reestimar fase
        for i in range(n_frames):
            start = i * hop_length
            if start + win_length <= len(audio):
                fft_vals = np.fft.rfft(audio[start:start + win_length] * window, n=n_fft)
                stft_matrix[:, i] = mag[:, i] * np.exp(1j * np.angle(fft_vals))

    # Síntesis final
    audio = np.zeros(expected_len)
    for i in range(n_frames):
        start = i * hop_length
        audio[start:start + win_length] += np.fft.irfft(stft_matrix[:, i], n=n_fft)[:win_length] * window
    # Compensación de solapamiento
    denom = np.zeros(expected_len)
    for i in range(n_frames):
        start = i * hop_length
        denom[start:start + win_length] += window ** 2
    audio = audio / np.maximum(denom, 1e-10)
    return audio


# -----------------------------------------------------------------------------
# Vocoder neuronal ligero tipo MelGAN / HiFi-GAN simplificado
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Bloque residual con dilataciones crecientes."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=d),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=d),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class NeuralVocoder(nn.Module):
    """
    Vocador neuronal que mapea mel-espectrogramas [batch, n_mels, frames]
    directamente a ondas [batch, samples].

    Inspirado en MelGAN/HiFi-GAN pero mucho más pequeño para entrenar desde
    cero con datasets reducidos.
    """

    def __init__(self,
                 n_mels: int = N_MELS,
                 hop_length: int = HOP_LENGTH,
                 channels: int = 256,
                 resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
                 upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)):
        super().__init__()
        self.hop_length = hop_length
        self.upsample_rates = upsample_rates
        total_upsample = int(np.prod(upsample_rates))
        if total_upsample != hop_length:
            raise ValueError(f"El producto de upsample_rates ({total_upsample}) debe ser igual a hop_length ({hop_length}).")

        # Proyección inicial de mel a espacio de características
        self.input_conv = nn.Conv1d(n_mels, channels, kernel_size=7, padding=3)

        # Upsampling progresivo con bloques residuales
        self.upsample_blocks = nn.ModuleList()
        in_ch = channels
        for rate in upsample_rates:
            out_ch = max(in_ch // 2, 64)
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=rate * 2, stride=rate, padding=rate // 2),
                    ResBlock(out_ch, kernel_size=3, dilations=(1, 3, 5)),
                )
            )
            in_ch = out_ch

        # Salida a 1 canal de audio
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [batch, n_mels, frames]
        x = self.input_conv(mel)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.output_conv(x)
        return x.squeeze(1)  # [batch, samples]


def spectrogram_loss(pred: torch.Tensor, target: torch.Tensor,
                     n_ffts: Tuple[int, ...] = (256, 512, 1024)) -> torch.Tensor:
    """
    Pérdida de espectrograma multi-escala (MSTFT) entre ondas predicha y target.
    """
    loss = 0.0
    for n_fft in n_ffts:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop,
                               win_length=n_fft, window=window,
                               return_complex=True, center=True)
        target_stft = torch.stft(target, n_fft=n_fft, hop_length=hop,
                                 win_length=n_fft, window=window,
                                 return_complex=True, center=True)
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        loss += F.l1_loss(pred_mag, target_mag)
    return loss / len(n_ffts)


def train_neural_vocoder(
    audio: np.ndarray,
    dsp: DSP,
    epochs: int = 50,
    batch_size: int = 8,
    segment_frames: int = 64,
) -> NeuralVocoder:
    """
    Entrena el vocoder neuronal con fragmentos del audio de referencia.
    Cada ejemplo es un par (mel, waveform) de segment_frames de largo.
    """
    print(f"\n[*] Entrenando vocoder neuronal en {DEVICE} por {epochs} épocas...")

    hop = dsp.hop_length
    samples_per_segment = segment_frames * hop

    # Generar pares mel/wave del audio completo
    pairs = []
    max_start = len(audio) - samples_per_segment - 1
    step = samples_per_segment // 2
    for start in range(0, max_start, step):
        wave_seg = audio[start:start + samples_per_segment]
        wave_seg = preemphasis(wave_seg, PREEMPHASIS)
        mel_seg = dsp.wav_to_mel_numpy(wave_seg)
        if mel_seg.shape[1] < segment_frames:
            continue
        mel_seg = mel_seg[:, :segment_frames]
        wave_seg = wave_seg[:samples_per_segment]
        pairs.append((mel_seg, wave_seg))

    if len(pairs) < batch_size:
        raise RuntimeError(f"No hay suficientes segmentos para entrenar el vocoder (encontrados {len(pairs)}).")

    print(f"    Pares mel/wave generados: {len(pairs)}")

    vocoder = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
    optimizer = torch.optim.AdamW(vocoder.parameters(), lr=2e-4, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    l1_criterion = nn.L1Loss()

    vocoder.train()
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        np.random.shuffle(pairs)
        losses = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            mels = torch.FloatTensor(np.stack([p[0] for p in batch_pairs])).to(DEVICE)
            waves = torch.FloatTensor(np.stack([p[1] for p in batch_pairs])).to(DEVICE)

            optimizer.zero_grad()
            pred_waves = vocoder(mels)

            # Recortar/empadronar para coincidir longitudes
            min_len = min(pred_waves.size(1), waves.size(1))
            pred_waves = pred_waves[:, :min_len]
            waves = waves[:, :min_len]

            loss_time = l1_criterion(pred_waves, waves)
            loss_spec = spectrogram_loss(pred_waves, waves)
            loss = loss_time + loss_spec

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vocoder.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = np.mean(losses)
        epoch_elapsed = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining_epochs = epochs - epoch
        eta = elapsed / epoch * remaining_epochs if epoch > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"    [{now}] Época vocoder [{epoch:03d}/{epochs}] -> loss: {mean_loss:.6f} "
                  f"| esta época: {epoch_elapsed:.1f}s | ETA total: {eta/60:.1f}min")

    print(f"[✔] Vocoder entrenado. Loss final: {mean_loss:.6f}")
    return vocoder


def neural_vocoder_infer(vocoder: NeuralVocoder, mel: np.ndarray) -> np.ndarray:
    """Genera audio a partir de un mel-espectrograma usando el vocoder neuronal."""
    vocoder.eval()
    mel_t = torch.FloatTensor(mel).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        wave = vocoder(mel_t)
    return wave.squeeze(0).cpu().numpy()


# -----------------------------------------------------------------------------
# Dataset y entrenamiento
# -----------------------------------------------------------------------------
class VoiceDataset:
    """Construye ejemplos (texto, mel, duraciones, pitch, energía) a partir de audio."""

    def __init__(self,
                 audio: np.ndarray,
                 phrases: List[str],
                 dsp: DSP,
                 max_seconds_per_chunk: float = 6.0):
        self.audio = audio
        self.phrases = phrases
        self.dsp = dsp
        self.samples_per_chunk = int(max_seconds_per_chunk * dsp.sr)
        self.examples = self._build_examples()

    def _build_examples(self):
        examples = []
        n_chunks = max(1, len(self.audio) // self.samples_per_chunk)
        for idx in range(n_chunks):
            start = idx * self.samples_per_chunk
            end = start + self.samples_per_chunk
            chunk = self.audio[start:end]
            if len(chunk) < self.dsp.hop_length * 4:
                continue

            # Preénfasis + mel
            chunk_pre = preemphasis(chunk, PREEMPHASIS)
            mel = self.dsp.wav_to_mel_numpy(chunk_pre)
            if mel.shape[1] < 4:
                continue

            phrase = self.phrases[idx % len(self.phrases)]
            tokens = text_to_sequence(phrase)
            durations = align_text_to_mel(tokens, mel)
            if sum(durations) > mel.shape[1]:
                # Escalar duraciones si DTW excede frames disponibles
                scale = mel.shape[1] / sum(durations)
                durations = [max(1, int(round(d * scale))) for d in durations]

            pitch = self.dsp.pitch(chunk_pre)
            energy = self.dsp.energy(chunk_pre)

            # Alinear pitch/energy al largo de mel por interpolación
            target_len = mel.shape[1]
            pitch = self._interp_to_len(pitch, target_len)
            energy = self._interp_to_len(energy, target_len)

            examples.append({
                "text": phrase,
                "tokens": tokens,
                "mel": mel,
                "durations": durations,
                "pitch": pitch,
                "energy": energy,
            })
        return examples

    @staticmethod
    def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        if len(arr) < 2:
            return np.zeros(target_len, dtype=np.float32)
        xp = np.linspace(0, target_len - 1, len(arr))
        x = np.arange(target_len)
        return np.interp(x, xp, arr).astype(np.float32)


def collate_batch(examples: List[dict], device: torch.device) -> dict:
    """Padding y tensores para un batch."""
    max_char_len = max(len(e["tokens"]) for e in examples)
    max_mel_len = max(e["mel"].shape[1] for e in examples)

    tokens = []
    durations = []
    mels = []
    pitches = []
    energies = []
    char_masks = []
    mel_masks = []

    for e in examples:
        t = e["tokens"]
        d = e["durations"]
        mel = e["mel"]
        p = e["pitch"]
        en = e["energy"]

        tokens.append(t + [PAD_IDX] * (max_char_len - len(t)))
        durations.append(d + [0] * (max_char_len - len(d)))

        mel_pad = np.zeros((mel.shape[0], max_mel_len - mel.shape[1]), dtype=np.float32)
        mels.append(np.concatenate([mel, mel_pad], axis=1))

        p_pad = np.zeros(max_mel_len - len(p), dtype=np.float32)
        pitches.append(np.concatenate([p, p_pad]))

        e_pad = np.zeros(max_mel_len - len(en), dtype=np.float32)
        energies.append(np.concatenate([en, e_pad]))

        char_masks.append([0] * len(t) + [1] * (max_char_len - len(t)))
        mel_masks.append([0] * mel.shape[1] + [1] * (max_mel_len - mel.shape[1]))

    return {
        "tokens": torch.LongTensor(tokens).to(device),
        "durations": torch.FloatTensor(durations).to(device),
        "mel": torch.FloatTensor(mels).to(device),
        "pitch": torch.FloatTensor(pitches).to(device),
        "energy": torch.FloatTensor(energies).to(device),
        "char_mask": torch.BoolTensor(char_masks).to(device),
        "mel_mask": torch.BoolTensor(mel_masks).to(device),
    }


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def _load_corpus(metadata_path: str = "dataset_metadata.json") -> List[str]:
    """Carga frases de entrenamiento desde metadata de subtítulos o usa default."""
    default_corpus = [
        "esta es una síntesis de voz limpia con tonos naturales",
        "inteligencia artificial corriendo nativa en la arquitectura metal",
        "el procesamiento de audio con redes neuronales recurrentes",
        "optimizando los tensores de pytorch para evitar el ruido de fondo",
        "la voz humana tiene matices prosódicos que debemos aprender",
        "entrenar desde cero permite controlar cada parámetro del sonido",
        "el pitch y la energía definen la expresividad de la voz",
    ]
    if not os.path.exists(metadata_path):
        return default_corpus
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frases = [item.get("texto", "").strip() for item in data if item.get("texto", "").strip()]
        if len(frases) >= 5:
            print(f"    Corpus cargado desde {metadata_path}: {len(frases)} frases.")
            return frases
    except Exception as exc:
        print(f"[!] No se pudo cargar {metadata_path}: {exc}")
    return default_corpus


MODEL_PATH = "voice_model.pt"
VOCODER_PATH = "voice_vocoder.pt"


def _build_dataset(audio_source: str, dsp: DSP) -> VoiceDataset:
    print("\n[*] Cargando audio de referencia...")
    audio = load_audio(audio_source, sr=SAMPLE_RATE)
    print(f"    Audio obtenido: {len(audio) / SAMPLE_RATE:.2f}s")

    corpus = _load_corpus("dataset_metadata.json")

    print("[*] Construyendo dataset con alineación DTW, pitch y energía...")
    dataset = VoiceDataset(audio, corpus, dsp)
    print(f"    Ejemplos generados: {len(dataset.examples)}")
    if len(dataset.examples) == 0:
        raise RuntimeError("No se pudieron generar ejemplos de entrenamiento.")
    return dataset


def _train_model(model: AcousticModel,
                 dataset: VoiceDataset,
                 epochs: int = 300,
                 batch_size: int = 4) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mel_criterion = nn.MSELoss()
    dur_criterion = nn.MSELoss()
    var_criterion = nn.MSELoss()

    print(f"[*] Entrenando en {DEVICE} por {epochs} épocas...")
    model.train()
    final_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        np.random.shuffle(dataset.examples)
        losses = []

        for i in range(0, len(dataset.examples), batch_size):
            batch = dataset.examples[i:i + batch_size]
            b = collate_batch(batch, DEVICE)

            optimizer.zero_grad()
            mel_pred, log_dur_pred, pitch_pred, energy_pred = model(
                b["tokens"],
                durations=b["durations"],
                target_mel_len=b["mel"].size(-1),
                pitch_target=b["pitch"],
                energy_target=b["energy"],
            )

            # Máscaras para ignorar padding
            mel_pred_T = mel_pred.transpose(1, 2)
            mel_mask = b["mel_mask"].unsqueeze(1)
            mel_loss = mel_criterion(mel_pred_T.masked_fill(mel_mask, 0.0), b["mel"].masked_fill(mel_mask, 0.0))

            log_dur_target = torch.log(b["durations"].float() + 1.0)
            dur_loss = dur_criterion(log_dur_pred.masked_fill(b["char_mask"], 0.0),
                                     log_dur_target.masked_fill(b["char_mask"], 0.0))

            pitch_char_target = []
            energy_char_target = []
            for ex_idx in range(len(batch)):
                durs = batch[ex_idx]["durations"]
                p = batch[ex_idx]["pitch"]
                e = batch[ex_idx]["energy"]
                ptr = 0
                p_chars = []
                e_chars = []
                for d in durs:
                    d = int(d)
                    p_chars.append(p[ptr:ptr + d].mean() if d > 0 else 0.0)
                    e_chars.append(e[ptr:ptr + d].mean() if d > 0 else 0.0)
                    ptr += d
                while len(p_chars) < b["tokens"].size(1):
                    p_chars.append(0.0)
                    e_chars.append(0.0)
                pitch_char_target.append(p_chars[:b["tokens"].size(1)])
                energy_char_target.append(e_chars[:b["tokens"].size(1)])

            pitch_char_target_t = torch.FloatTensor(pitch_char_target).to(DEVICE)
            energy_char_target_t = torch.FloatTensor(energy_char_target).to(DEVICE)

            pitch_loss = var_criterion(pitch_pred.masked_fill(b["char_mask"], 0.0),
                                       pitch_char_target_t.masked_fill(b["char_mask"], 0.0))
            energy_loss = var_criterion(energy_pred.masked_fill(b["char_mask"], 0.0),
                                        energy_char_target_t.masked_fill(b["char_mask"], 0.0))

            loss = mel_loss + dur_loss + 0.1 * pitch_loss + 0.1 * energy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = np.mean(losses)
        final_loss = mean_loss
        epoch_elapsed = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining_epochs = epochs - epoch
        eta = elapsed / epoch * remaining_epochs if epoch > 0 else 0.0

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"    [{now}] Época [{epoch:04d}/{epochs}] -> loss: {mean_loss:.6f} "
                  f"| esta época: {epoch_elapsed:.1f}s | ETA total: {eta/60:.1f}min")

    return final_loss


def _mel_stats(dataset: VoiceDataset) -> Tuple[float, float]:
    mel_mean = np.mean([e["mel"].mean() for e in dataset.examples])
    mel_std = np.mean([e["mel"].std() for e in dataset.examples])
    return float(mel_mean), float(mel_std)


def synthesize(model: AcousticModel,
               vocoder: Optional[NeuralVocoder],
               dsp: DSP,
               phrase: str,
               output_wav: str,
               mel_mean: float,
               mel_std: float) -> str:
    """Sintetiza una frase usando un modelo ya entrenado."""
    print(f"\n[*] Sintetizando: '{phrase}'")
    model.eval()
    tokens = torch.LongTensor([text_to_sequence(phrase)]).to(DEVICE)

    with torch.no_grad():
        mel_pred, _, _, _ = model(tokens)

    mel_pred_np = mel_pred.squeeze(0).cpu().numpy()

    # Normalización inversa basada en estadísticas del dataset
    mel_pred_np = (mel_pred_np - mel_pred_np.mean()) / (mel_pred_np.std() + 1e-8)
    mel_pred_np = mel_pred_np * mel_std + mel_mean

    if vocoder is not None:
        print("[*] Reconstruyendo onda con vocoder neuronal...")
        audio_out = neural_vocoder_infer(vocoder, mel_pred_np)
        audio_out = inv_preemphasis(audio_out)
    else:
        print("[*] Reconstruyendo onda con vocoder Griffin-Lim (fallback)...")
        mag = mel_to_linear(mel_pred_np, dsp.mel_filters_t.cpu().numpy())
        audio_out = griffin_lim(mag, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_iter=60)
        audio_out = inv_preemphasis(audio_out)

    audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95

    write_wav(output_wav, audio_out, sr=SAMPLE_RATE)
    print(f"[✔] Archivo guardado: {output_wav} ({len(audio_out) / SAMPLE_RATE:.2f}s)")
    return output_wav


def _evaluate_model(model: AcousticModel, dataset: VoiceDataset) -> float:
    """Calcula el MSE promedio de mel en una pasada de validación sobre el dataset."""
    model.eval()
    mel_criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for example in dataset.examples[:50]:  # usamos hasta 50 ejemplos para no tardar demasiado
            tokens = torch.LongTensor([example["tokens"]]).to(DEVICE)
            target_mel = torch.FloatTensor(example["mel"]).unsqueeze(0).to(DEVICE)
            mel_pred, _, _, _ = model(tokens, target_mel_len=target_mel.size(-1))
            mel_pred_T = mel_pred.transpose(1, 2)
            loss = mel_criterion(mel_pred_T, target_mel)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train_and_synthesize(audio_source: str,
                         output_wav: str = "voz_generada.wav",
                         test_phrase: str = "esta es una síntesis de voz limpia con tonos naturales",
                         epochs: int = 300,
                         batch_size: int = 4,
                         vocoder_epochs: int = 50,
                         force_retrain: bool = False):
    dsp = DSP()
    dataset = _build_dataset(audio_source, dsp)
    mel_mean, mel_std = _mel_stats(dataset)

    model = AcousticModel().to(DEVICE)
    vocoder: Optional[NeuralVocoder] = None

    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"[*] Cargando modelo acústico desde '{MODEL_PATH}'...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("[✔] Modelo acústico cargado. Omitiendo entrenamiento acústico.")
    else:
        final_loss = _train_model(model, dataset, epochs=epochs, batch_size=batch_size)
        print(f"\n[✔] Entrenamiento acústico finalizado. Loss final: {final_loss:.6f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[✔] Modelo acústico guardado en '{MODEL_PATH}'")

        val_mse = _evaluate_model(model, dataset)
        print(f"    MSE promedio de mel (validación interna): {val_mse:.6f}")

    if os.path.exists(VOCODER_PATH) and not force_retrain:
        print(f"[*] Cargando vocoder desde '{VOCODER_PATH}'...")
        vocoder = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
        vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=DEVICE))
        print("[✔] Vocoder cargado. Omitiendo entrenamiento del vocoder.")
    else:
        vocoder = train_neural_vocoder(dataset.audio, dsp, epochs=vocoder_epochs)
        torch.save(vocoder.state_dict(), VOCODER_PATH)
        print(f"[✔] Vocoder guardado en '{VOCODER_PATH}'")

    synthesize(model, vocoder, dsp, test_phrase, output_wav, mel_mean=mel_mean, mel_std=mel_std)


def synthesize_only(phrase: str,
                    output_wav: str = "voz_generada.wav",
                    audio_source: str = "audio_referencia.wav"):
    """Sintetiza una frase usando los modelos guardados, sin entrenar."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No existe '{MODEL_PATH}'. Entrena primero con train_and_synthesize()."
        )

    dsp = DSP()
    dataset = _build_dataset(audio_source, dsp)
    mel_mean, mel_std = _mel_stats(dataset)

    model = AcousticModel().to(DEVICE)
    print(f"[*] Cargando modelo acústico desde '{MODEL_PATH}'...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    vocoder: Optional[NeuralVocoder] = None
    if os.path.exists(VOCODER_PATH):
        print(f"[*] Cargando vocoder desde '{VOCODER_PATH}'...")
        vocoder = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
        vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=DEVICE))
        print("[✔] Vocoder cargado.")
    else:
        print("[!] No se encontró vocoder entrenado. Se usará Griffin-Lim como fallback.")

    synthesize(model, vocoder, dsp, phrase, output_wav, mel_mean=mel_mean, mel_std=mel_std)


if __name__ == "__main__":
    AUDIO_FUENTE = "audio_referencia.wav" if os.path.exists("audio_referencia.wav") else "https://youtu.be/bhQnooudZcs?si=WLtSJNfCwZ1EZyQD"
    FRASE_TEST = "esta es una síntesis de voz limpia con tonos naturales corriendo en mi procesador M4"
    train_and_synthesize(AUDIO_FUENTE, test_phrase=FRASE_TEST)
