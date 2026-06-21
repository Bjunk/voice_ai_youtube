"""
voice.py — Síntesis de voz entrenada desde cero con alineación real por timestamps.

Mejoras clave sobre la versión anterior:
  ✦ Dataset construido con timestamps reales de dataset_metadata.json.
    El audio del segmento [inicio:fin] se empareja con el texto exacto de ese
    segmento — elimina el problema raíz de la versión anterior (round-robin ciego).
  ✦ Normalización de mel espectrogramas (media/std del dataset completo).
  ✦ Pérdida L1 en mel (más robusta que MSE para espectrogramas).
  ✦ LR warmup + CosineAnnealing para convergencia estable.
  ✦ Log detallado por componente de pérdida.
  ✦ Vocoder neuronal entrenado sobre los mismos segmentos alineados.

Sobre los epochs:
  FastSpeech 2 original usa ~160 000 steps sobre LJSpeech (24 h de estudio).
  Con este dataset (~200 segmentos, batch 4 → 50 steps/epoch):
    160 000 steps ≈ 3 200 epochs equivalentes.
  Con datos de YouTube (ruidosos), 500–1000 epochs producen audio inteligible.
  La versión anterior no convergía en 300 epochs porque los datos eran ruido puro
  (texto aleatorio vs. audio sin relación). Con alineación real, la convergencia
  ocurre mucho antes.

Requiere: numpy, torch, scipy (opcional, mejora DTW y filtros)
"""

import json
import math
import os
import struct
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.spatial.distance import cdist
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[*] Dispositivo de aceleración: {DEVICE}")

SAMPLE_RATE   = 22050
N_FFT         = 1024
HOP_LENGTH    = 256
WIN_LENGTH    = 1024
N_MELS        = 80
MEL_FMIN      = 0.0
MEL_FMAX      = 8000.0
MAX_WAV_VALUE = 32768.0
PREEMPHASIS   = 0.97

MODEL_PATH   = "voice_model.pt"
VOCODER_PATH = "voice_vocoder.pt"
STATS_PATH   = "mel_stats.json"


# ─── Utilidades de audio ──────────────────────────────────────────────────────

def _which(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def load_audio(source: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Carga audio desde URL (yt-dlp) o ruta local.
    Devuelve float32 mono normalizado al sample rate indicado.
    """
    source = source.strip()
    es_url = source.startswith(("http://", "https://"))

    if es_url and not _which("yt-dlp"):
        raise RuntimeError("Se requiere 'yt-dlp' en el PATH.")
    if not _which("ffmpeg"):
        raise RuntimeError("Se requiere 'ffmpeg' en el PATH.")

    tmp_path = tempfile.mktemp(suffix=".raw")
    try:
        if es_url:
            p1 = subprocess.Popen(
                ["yt-dlp", "-f", "bestaudio", "--audio-quality", "0", "-o", "-", source],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            ffmpeg_input = ["-i", "pipe:0"]
            stdin = p1.stdout
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"No se encontró: {source}")
            p1 = None
            ffmpeg_input = ["-i", source]
            stdin = None

        p2 = subprocess.Popen(
            ["ffmpeg", "-y", *ffmpeg_input, "-vn", "-f", "s16le",
             "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", tmp_path],
            stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
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


def preemphasis(wave: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    if HAS_SCIPY:
        return lfilter([1.0, -coeff], [1.0], wave)
    out = np.zeros_like(wave)
    out[0] = wave[0]
    out[1:] = wave[1:] - coeff * wave[:-1]
    return out


def inv_preemphasis(wave: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    if HAS_SCIPY:
        return lfilter([1.0], [1.0, -coeff], wave)
    out = np.zeros_like(wave)
    out[0] = wave[0]
    for i in range(1, len(wave)):
        out[i] = wave[i] + coeff * out[i - 1]
    return out


def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    audio = np.clip(audio, -1.0, 1.0)
    pcm   = (audio * 32767.0).astype(np.int16)
    data  = pcm.tobytes()
    br    = sr * 2
    hdr   = struct.pack("<4sI4s4sIHHIIHH4sI",
                        b"RIFF", 36 + len(data), b"WAVE", b"fmt ",
                        16, 1, 1, sr, br, 2, 16, b"data", len(data))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data)


# ─── Procesamiento de texto ───────────────────────────────────────────────────

ALPHABET = (
    "_"   # padding
    " "
    "abcdefghijklmnopqrstuvwxyz"
    "ñ"
    "áéíóúü"
    ".,!?-"
)
PAD_IDX = 0
_DIGIT_WORDS = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
}


def text_to_sequence(text: str) -> List[int]:
    seq = []
    for ch in text.lower():
        if ch in ALPHABET:
            seq.append(ALPHABET.index(ch))
        elif ch.isdigit():
            seq.extend(text_to_sequence(_DIGIT_WORDS.get(ch, "")))
    return seq if seq else [ALPHABET.index(" ")]


# ─── DSP ──────────────────────────────────────────────────────────────────────

def _mel_scale(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _inv_mel_scale(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filter_bank(sr, n_fft, n_mels, f_min, f_max) -> np.ndarray:
    low_mel  = _mel_scale(max(f_min, 1.0))
    high_mel = _mel_scale(f_max)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = np.array([_inv_mel_scale(m) for m in mel_pts])
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    filt     = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            k = np.arange(l, c)
            filt[m - 1, l:c] = (k - l) / (c - l)
        if r > c:
            k = np.arange(c, r)
            filt[m - 1, c:r] = (r - k) / (r - c)
    return filt


class DSP:
    def __init__(self, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 win_length=WIN_LENGTH, n_mels=N_MELS,
                 f_min=MEL_FMIN, f_max=MEL_FMAX):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels     = n_mels
        mel_filt            = build_mel_filter_bank(sr, n_fft, n_mels, f_min, f_max)
        self.mel_filters    = mel_filt
        self.mel_filters_t  = torch.from_numpy(mel_filt).float().to(DEVICE)
        self.window_t       = torch.hann_window(win_length).to(DEVICE)

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        pad = (self.n_fft - self.hop_length) // 2
        wav = F.pad(wav, (pad, pad), mode="reflect")
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window_t,
                          return_complex=True, center=False)
        power = torch.abs(stft) ** 2
        mel   = torch.matmul(self.mel_filters_t, power)
        return torch.log(mel.clamp(min=1e-5))

    def wav_to_mel_numpy(self, wav: np.ndarray) -> np.ndarray:
        wav_t = torch.from_numpy(wav).float().to(DEVICE)
        with torch.no_grad():
            mel = self.wav_to_mel(wav_t)
        return mel.squeeze(0).cpu().numpy()

    def pitch(self, wav: np.ndarray) -> np.ndarray:
        """F0 por frame mediante YIN simplificado (Hz; 0 = no vozeado)."""
        hop, win   = self.hop_length, self.win_length
        min_p      = max(1, int(self.sr / 500.0))
        max_p      = min(win // 2, int(self.sr / 50.0))
        n_frames   = max(0, 1 + (len(wav) - win) // hop)
        f0         = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            frame = wav[i * hop: i * hop + win]
            frame = (frame - frame.mean()) * np.hanning(win)
            diff  = np.zeros(max_p, dtype=np.float32)
            for tau in range(1, max_p):
                diff[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2)
            cmndf = np.zeros(max_p, dtype=np.float32)
            cum   = 0.0
            for tau in range(1, max_p):
                cum += diff[tau]
                cmndf[tau] = diff[tau] / ((cum / tau) + 1e-12)
            best = 0
            for tau in range(min_p, max_p - 1):
                if cmndf[tau] < 0.1 and cmndf[tau] < cmndf[tau-1] and cmndf[tau] < cmndf[tau+1]:
                    best = tau
                    break
            if best > 0:
                y1, y2, y3 = cmndf[best-1], cmndf[best], cmndf[best+1]
                d = 2.0 * (y1 - 2*y2 + y3)
                if abs(d) > 1e-12:
                    best = best + (y1 - y3) / d
                f0[i] = self.sr / best
        return f0

    def energy(self, wav: np.ndarray) -> np.ndarray:
        hop, win = self.hop_length, self.win_length
        n_frames = max(0, 1 + (len(wav) - win) // hop)
        en       = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            frame  = wav[i*hop: i*hop+win] * np.hanning(win)
            en[i]  = np.sqrt(np.mean(frame ** 2) + 1e-12)
        return en


# ─── Alineación texto-audio con DTW ──────────────────────────────────────────

def _dtw(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    if HAS_SCIPY:
        cost = cdist(x, y, metric="euclidean")
    else:
        cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

    n, m   = cost.shape
    acc    = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i-1, j-1] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])

    path_y = np.zeros(n, dtype=int)
    i, j   = n, m
    while i > 0 and j > 0:
        path_y[i - 1] = j - 1
        choices = {
            (i-1, j-1): acc[i-1, j-1],
            (i-1, j):   acc[i-1, j],
            (i,   j-1): acc[i,   j-1],
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
    Alinea caracteres a frames de mel.
    Con timestamps reales, el mel ya corresponde al texto → DTW es significativo.
    """
    n_chars  = len(chars)
    n_frames = mel.shape[1]
    if n_chars == 0 or n_frames == 0:
        return [1] * max(n_chars, 1)

    # Distribución uniforme como ancla base (más estable que DTW puro en datos ruidosos)
    base_dur = n_frames / n_chars
    if base_dur < 1:
        return [1] * n_chars

    # Features de carácter
    char_feats = np.zeros((n_chars, 8), dtype=np.float32)
    for i, c in enumerate(chars):
        char_feats[i, 0] = c / len(ALPHABET)
        char_feats[i, 1] = i / max(n_chars - 1, 1)
        char_feats[i, 2] = math.sin(i * 0.3)
        char_feats[i, 3] = math.cos(i * 0.3)

    # Reducir mel a 8 grupos para comparar con char_feats
    mel_T   = mel.T.astype(np.float32)
    groups  = 8
    bpg     = mel_T.shape[1] // groups
    mel_red = np.zeros((n_frames, groups), dtype=np.float32)
    for g in range(groups):
        s = g * bpg
        e = s + bpg if g < groups - 1 else mel_T.shape[1]
        mel_red[:, g] = mel_T[:, s:e].mean(axis=1)

    mel_norm = (mel_red - mel_red.mean(0)) / (mel_red.std(0) + 1e-8)

    path, _ = _dtw(char_feats, mel_norm)
    durs    = [0] * n_chars
    for fi in path:
        if 0 <= fi < n_chars:
            durs[fi] += 1

    # Suavizar: mezclar DTW con distribución uniforme (30% uniforme, 70% DTW)
    uniform = [max(1, int(round(base_dur)))] * n_chars
    durs    = [max(1, int(round(0.7 * d + 0.3 * u))) for d, u in zip(durs, uniform)]

    # Ajustar suma total a n_frames
    total = sum(durs)
    if total != n_frames:
        diff = n_frames - total
        durs[-1] = max(1, durs[-1] + diff)
    return durs


# ─── Modelo acústico FastSpeech 2 ────────────────────────────────────────────

def positional_encoding(length: int, channels: int, device) -> torch.Tensor:
    pos     = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div     = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / channels)
    )
    pe = torch.zeros(length, channels, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FFTBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        pad          = kernel_size // 2
        self.conv1   = nn.Conv1d(channels, channels * 4, kernel_size, padding=pad)
        self.conv2   = nn.Conv1d(channels * 4, channels, kernel_size, padding=pad)
        self.ln      = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        x   = x.transpose(1, 2)
        x   = F.relu(self.dropout(self.conv1(x)))
        x   = self.dropout(self.conv2(x))
        x   = self.ln(x.transpose(1, 2) + res)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class LengthRegulator(nn.Module):
    def forward(self, x: torch.Tensor, durations: torch.Tensor,
                max_len: Optional[int] = None) -> torch.Tensor:
        output = []
        for b in range(x.size(0)):
            exp = []
            for i in range(x.size(1)):
                r = max(1, int(torch.round(durations[b, i]).item()))
                exp.append(x[b, i:i+1].repeat(r, 1))
            output.append(torch.cat(exp, dim=0))

        ml = max(o.size(0) for o in output)
        if max_len is not None:
            ml = max(ml, max_len)
        padded = []
        for o in output:
            if o.size(0) < ml:
                pad = torch.zeros(ml - o.size(0), o.size(1), device=o.device)
                o   = torch.cat([o, pad], dim=0)
            else:
                o = o[:ml]
            padded.append(o)
        return torch.stack(padded, dim=0)


class VariancePredictor(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        p          = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=p)
        self.ln1   = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=p)
        self.ln2   = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        h   = F.relu(self.ln1(self.conv1(x.transpose(1,2)).transpose(1,2)))
        h   = self.drop(h)
        h   = F.relu(self.ln2(self.conv2(h.transpose(1,2)).transpose(1,2)))
        h   = self.drop(h) + res
        out = self.proj(h).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class AcousticModel(nn.Module):
    """
    Modelo TTS basado en FastSpeech 2.
    Predice mel-espectrograma normalizado, duración, pitch y energía desde texto.
    """
    def __init__(self, vocab_size: int = len(ALPHABET), channels: int = 256,
                 n_fft_blocks: int = 4, mel_dim: int = N_MELS,
                 dropout: float = 0.1):
        super().__init__()
        self.channels  = channels
        self.mel_dim   = mel_dim
        self.embedding = nn.Embedding(vocab_size, channels, padding_idx=PAD_IDX)

        self.encoder = nn.ModuleList([
            FFTBlock(channels, dropout=dropout) for _ in range(n_fft_blocks)
        ])

        self.dur_pred    = VariancePredictor(channels)
        self.pitch_pred  = VariancePredictor(channels)
        self.energy_pred = VariancePredictor(channels)

        self.pitch_bins   = nn.Parameter(
            torch.linspace(math.log(50.0), math.log(600.0), 256).exp(),
            requires_grad=False)
        self.pitch_emb    = nn.Embedding(256, channels)
        self.energy_bins  = nn.Parameter(
            torch.linspace(0.0, 1.0, 256), requires_grad=False)
        self.energy_emb   = nn.Embedding(256, channels)

        self.length_reg = LengthRegulator()

        self.decoder = nn.ModuleList([
            FFTBlock(channels, dropout=dropout) for _ in range(n_fft_blocks)
        ])
        self.mel_proj = nn.Linear(channels, mel_dim)

    def _quantize(self, v: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.abs(v.unsqueeze(-1) - bins.view(1, 1, -1)), dim=-1)

    def _expand_var(self, var: torch.Tensor, durs: torch.Tensor,
                    max_len: int) -> torch.Tensor:
        out = []
        for b in range(var.size(0)):
            exp = []
            for i in range(var.size(1)):
                r = max(1, int(torch.round(durs[b, i]).item()))
                exp.append(var[b, i].repeat(r))
            exp = torch.cat(exp, dim=0)
            if exp.size(0) < max_len:
                exp = torch.cat([exp, torch.zeros(max_len - exp.size(0), device=exp.device)])
            else:
                exp = exp[:max_len]
            out.append(exp)
        return torch.stack(out, dim=0)

    def _pad_or_trunc(self, t: torch.Tensor, ml: int) -> torch.Tensor:
        b, l = t.shape
        if l < ml:
            return torch.cat([t, torch.zeros(b, ml - l, device=t.device)], dim=1)
        return t[:, :ml]

    def forward(self, tokens: torch.Tensor,
                durations: Optional[torch.Tensor] = None,
                target_mel_len: Optional[int] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None):
        x    = self.embedding(tokens) * math.sqrt(self.channels)
        bsz, clen, _ = x.shape
        x    = x + positional_encoding(clen, self.channels, x.device)
        mask = (tokens == PAD_IDX)

        for layer in self.encoder:
            x = layer(x, mask=mask)

        log_dur_pred   = self.dur_pred(x, mask=mask)
        pitch_pred_chr = self.pitch_pred(x, mask=mask)
        energy_pred_chr = self.energy_pred(x, mask=mask)

        if durations is not None:
            durs_for_reg = durations.float()
        else:
            durs_for_reg = torch.clamp(torch.exp(log_dur_pred) - 1.0, min=1.0)

        x_exp   = self.length_reg(x, durs_for_reg, max_len=target_mel_len)
        mel_len = x_exp.size(1)

        if pitch_target is not None and energy_target is not None:
            p_frame = self._pad_or_trunc(pitch_target,  mel_len)
            e_frame = self._pad_or_trunc(energy_target, mel_len)
        else:
            p_frame = self._expand_var(pitch_pred_chr,  durs_for_reg, mel_len)
            e_frame = self._expand_var(energy_pred_chr, durs_for_reg, mel_len)

        p_idx  = self._quantize(p_frame, self.pitch_bins)
        e_idx  = self._quantize(e_frame, self.energy_bins)
        x_exp  = x_exp + self.pitch_emb(p_idx) + self.energy_emb(e_idx)
        x_exp  = x_exp + positional_encoding(mel_len, self.channels, x_exp.device)

        for layer in self.decoder:
            x_exp = layer(x_exp)

        mel_pred = self.mel_proj(x_exp)
        return mel_pred, log_dur_pred, pitch_pred_chr, energy_pred_chr


# ─── Vocoder: inversión mel + Griffin-Lim (fallback) ─────────────────────────

def mel_to_linear(mel: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    return np.maximum(np.dot(np.linalg.pinv(mel_filters), np.exp(mel)), 1e-5)


def griffin_lim(mag: np.ndarray, n_fft=N_FFT, hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH, n_iter=60) -> np.ndarray:
    nf  = mag.shape[1]
    el  = nf * hop_length + win_length
    ang = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    S   = mag * ang
    win = np.hanning(win_length)

    for _ in range(n_iter):
        a = np.zeros(el)
        for i in range(nf):
            a[i*hop_length: i*hop_length+win_length] += (
                np.fft.irfft(S[:, i], n=n_fft)[:win_length] * win)
        for i in range(nf):
            s = i * hop_length
            if s + win_length <= len(a):
                fv = np.fft.rfft(a[s: s+win_length] * win, n=n_fft)
                S[:, i] = mag[:, i] * np.exp(1j * np.angle(fv))

    a = np.zeros(el)
    d = np.zeros(el)
    for i in range(nf):
        s = i * hop_length
        a[s: s+win_length] += np.fft.irfft(S[:, i], n=n_fft)[:win_length] * win
        d[s: s+win_length] += win ** 2
    return a / np.maximum(d, 1e-10)


# ─── Vocoder neuronal (MelGAN ligero) ─────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int, ks: int = 3, dils: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(ch, ch, ks, padding=(ks-1)*d//2, dilation=d),
                nn.LeakyReLU(0.2),
                nn.Conv1d(ch, ch, ks, padding=(ks-1)*d//2, dilation=d),
            ) for d in dils
        ])

    def forward(self, x):
        for c in self.convs:
            x = x + c(x)
        return x


class NeuralVocoder(nn.Module):
    def __init__(self, n_mels=N_MELS, hop_length=HOP_LENGTH, channels=256,
                 upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)):
        super().__init__()
        assert math.prod(upsample_rates) == hop_length, (
            f"prod(upsample_rates) debe ser {hop_length}"
        )
        self.inp  = nn.Conv1d(n_mels, channels, 7, padding=3)
        self.ups  = nn.ModuleList()
        ch        = channels
        for r in upsample_rates:
            och = max(ch // 2, 64)
            self.ups.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(ch, och, r*2, stride=r, padding=r//2),
                ResBlock(och),
            ))
            ch = och
        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.inp(mel)
        for b in self.ups:
            x = b(x)
        return self.out(x).squeeze(1)


def _spectrogram_loss(pred: torch.Tensor, target: torch.Tensor,
                      n_ffts: Tuple[int, ...] = (256, 512, 1024)) -> torch.Tensor:
    loss = torch.tensor(0.0, device=pred.device)
    for nf in n_ffts:
        hop = nf // 4
        win = torch.hann_window(nf, device=pred.device)
        ps  = torch.stft(pred,   nf, hop, nf, win, return_complex=True, center=True)
        ts  = torch.stft(target, nf, hop, nf, win, return_complex=True, center=True)
        loss = loss + F.l1_loss(torch.abs(ps), torch.abs(ts))
    return loss / len(n_ffts)


def train_neural_vocoder(audio: np.ndarray, dsp: DSP,
                          epochs: int = 100, batch_size: int = 8,
                          segment_frames: int = 64) -> NeuralVocoder:
    print(f"\n[*] Entrenando vocoder neuronal en {DEVICE} por {epochs} épocas...")
    hop      = dsp.hop_length
    seg_len  = segment_frames * hop
    pairs    = []
    step     = seg_len // 2
    for start in range(0, max(0, len(audio) - seg_len), step):
        w = preemphasis(audio[start: start + seg_len])
        m = dsp.wav_to_mel_numpy(w)
        if m.shape[1] < segment_frames:
            continue
        pairs.append((m[:, :segment_frames], w[:seg_len]))

    if len(pairs) < batch_size:
        raise RuntimeError(f"Pocos segmentos para vocoder ({len(pairs)}).")

    print(f"    Segmentos disponibles: {len(pairs)}")
    voc  = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
    opt  = torch.optim.AdamW(voc.parameters(), lr=2e-4, betas=(0.8, 0.99))
    sch  = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    l1   = nn.L1Loss()
    t0   = time.time()
    loss_val = 0.0

    for ep in range(1, epochs + 1):
        np.random.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs), batch_size):
            bp   = pairs[i: i + batch_size]
            mels = torch.FloatTensor(np.stack([p[0] for p in bp])).to(DEVICE)
            wavs = torch.FloatTensor(np.stack([p[1] for p in bp])).to(DEVICE)
            opt.zero_grad()
            pw   = voc(mels)
            ml   = min(pw.size(1), wavs.size(1))
            loss = l1(pw[:, :ml], wavs[:, :ml]) + _spectrogram_loss(pw[:, :ml], wavs[:, :ml])
            loss.backward()
            nn.utils.clip_grad_norm_(voc.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sch.step()
        loss_val = np.mean(losses)
        if ep % 25 == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - t0
            eta     = elapsed / ep * (epochs - ep)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Vocoder [{ep:03d}/{epochs}] loss={loss_val:.4f} "
                  f"| ETA {eta/60:.1f}min")

    print(f"[✔] Vocoder entrenado. Loss final: {loss_val:.6f}")
    return voc


# ─── Dataset con timestamps reales ───────────────────────────────────────────

class VoiceDataset:
    """
    Construye ejemplos (tokens, mel_norm, durations, pitch, energy) usando
    los timestamps reales de dataset_metadata.json.

    La mejora crítica respecto a la versión anterior: el audio del segmento
    [inicio:fin] corresponde exactamente al texto de ese subtítulo.
    """

    def __init__(self, audio: np.ndarray, metadata: List[dict], dsp: DSP,
                 mel_mean: float = 0.0, mel_std: float = 1.0):
        self.audio    = audio
        self.metadata = metadata
        self.dsp      = dsp
        self.mel_mean = mel_mean
        self.mel_std  = mel_std
        self.examples = self._build()

    def _build(self) -> List[dict]:
        examples = []
        sr       = self.dsp.sr

        for entry in self.metadata:
            texto  = entry.get("texto", "").strip()
            inicio = float(entry.get("inicio", 0.0))
            fin    = float(entry.get("fin",    0.0))

            if not texto or fin <= inicio + 0.05:
                continue

            ss  = int(inicio * sr)
            es  = min(int(fin * sr), len(self.audio))
            seg = self.audio[ss:es]

            if len(seg) < self.dsp.win_length * 4:
                continue

            seg_pre = preemphasis(seg)
            mel     = self.dsp.wav_to_mel_numpy(seg_pre)   # [n_mels, frames]
            if mel.shape[1] < 4:
                continue

            tokens  = text_to_sequence(texto)
            if not tokens:
                continue
            durs    = align_text_to_mel(tokens, mel)

            # Asegurar que la suma de duraciones no exceda los frames
            total = sum(durs)
            if total > mel.shape[1]:
                scale = mel.shape[1] / total
                durs  = [max(1, int(round(d * scale))) for d in durs]
                # Ajuste fino
                diff  = mel.shape[1] - sum(durs)
                durs[-1] = max(1, durs[-1] + diff)

            pitch  = self.dsp.pitch(seg_pre)
            energy = self.dsp.energy(seg_pre)

            def interp(arr, tgt):
                if len(arr) == tgt:
                    return arr
                if len(arr) < 2:
                    return np.zeros(tgt, dtype=np.float32)
                xp = np.linspace(0, tgt - 1, len(arr))
                return np.interp(np.arange(tgt), xp, arr).astype(np.float32)

            tgt_len = mel.shape[1]
            pitch   = interp(pitch,  tgt_len)
            energy  = interp(energy, tgt_len)

            # Normalizar mel
            mel_norm = (mel - self.mel_mean) / (self.mel_std + 1e-8)

            examples.append({
                "text":     texto,
                "tokens":   tokens,
                "mel":      mel_norm,
                "mel_raw":  mel,
                "durations": durs,
                "pitch":    pitch,
                "energy":   energy,
            })
        return examples


def _compute_mel_stats(examples: List[dict]) -> Tuple[float, float]:
    """Computa media y std sobre todos los mel del dataset (para normalización)."""
    all_mel = np.concatenate([e["mel_raw"].flatten() for e in examples])
    return float(all_mel.mean()), float(all_mel.std() + 1e-8)


def _collate(examples: List[dict], device) -> dict:
    max_char = max(len(e["tokens"]) for e in examples)
    max_mel  = max(e["mel"].shape[1] for e in examples)

    tokens = []; durs = []; mels = []; pitches = []; energies = []
    cmasks = []; mmasks = []

    for e in examples:
        t, d, m, p, en = (e["tokens"], e["durations"], e["mel"],
                          e["pitch"], e["energy"])
        tokens.append(t  + [PAD_IDX] * (max_char - len(t)))
        durs.append(d    + [0]        * (max_char - len(d)))
        mp  = np.zeros((m.shape[0], max_mel - m.shape[1]), dtype=np.float32)
        mels.append(np.concatenate([m, mp], axis=1))
        pp  = np.zeros(max_mel - len(p),  dtype=np.float32)
        ep  = np.zeros(max_mel - len(en), dtype=np.float32)
        pitches.append(np.concatenate([p, pp]))
        energies.append(np.concatenate([en, ep]))
        cmasks.append([False] * len(t) + [True] * (max_char - len(t)))
        mmasks.append([False] * m.shape[1] + [True] * (max_mel - m.shape[1]))

    return {
        "tokens":   torch.LongTensor(tokens).to(device),
        "durations": torch.FloatTensor(durs).to(device),
        "mel":      torch.FloatTensor(np.array(mels)).to(device),
        "pitch":    torch.FloatTensor(np.array(pitches)).to(device),
        "energy":   torch.FloatTensor(np.array(energies)).to(device),
        "char_mask": torch.BoolTensor(cmasks).to(device),
        "mel_mask":  torch.BoolTensor(mmasks).to(device),
    }


# ─── Entrenamiento del modelo acústico ───────────────────────────────────────

def _warmup_cosine_schedule(optimizer, warmup_epochs: int, total_epochs: int,
                             base_lr: float, min_lr: float = 1e-5):
    """LR warmup lineal → cosine decay."""
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_model(model: AcousticModel, dataset: VoiceDataset,
                 epochs: int = 500, batch_size: int = 4) -> float:
    base_lr  = 3e-4
    opt      = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    warmup   = max(10, epochs // 20)
    sched    = _warmup_cosine_schedule(opt, warmup, epochs, base_lr)

    l1_loss  = nn.L1Loss()
    mse_loss = nn.MSELoss()

    print(f"[*] Entrenando en {DEVICE} por {epochs} épocas "
          f"({len(dataset.examples)} ejemplos, batch={batch_size})...")
    print(f"    LR inicial: {base_lr}  |  Warmup: {warmup} épocas")

    model.train()
    t0         = time.time()
    final_loss = float("inf")

    for ep in range(1, epochs + 1):
        ep_t = time.time()
        np.random.shuffle(dataset.examples)
        losses = {"total": [], "mel": [], "dur": [], "pitch": [], "energy": []}

        for i in range(0, len(dataset.examples), batch_size):
            b   = _collate(dataset.examples[i: i + batch_size], DEVICE)
            opt.zero_grad()

            mel_pred, logdur_pred, pitch_pred, energy_pred = model(
                b["tokens"],
                durations       = b["durations"],
                target_mel_len  = b["mel"].size(-1),
                pitch_target    = b["pitch"],
                energy_target   = b["energy"],
            )

            # Alinear longitud de mel_pred con el target (el LengthRegulator
            # puede diferir ±pocos frames por redondeo de duraciones).
            tgt_len  = b["mel"].size(-1)
            pred_len = mel_pred.size(1)
            if pred_len > tgt_len:
                mel_pred = mel_pred[:, :tgt_len, :]
            elif pred_len < tgt_len:
                pad      = torch.zeros(mel_pred.size(0), tgt_len - pred_len,
                                       mel_pred.size(2), device=mel_pred.device)
                mel_pred = torch.cat([mel_pred, pad], dim=1)

            # Mel: L1 sobre frames no enmascarados (normalizado)
            mm    = b["mel_mask"].unsqueeze(1)
            mel_l = l1_loss(
                mel_pred.transpose(1, 2).masked_fill(mm, 0.0),
                b["mel"].masked_fill(mm, 0.0),
            )

            # Duración: MSE en espacio log
            log_dur_tgt = torch.log(b["durations"].float() + 1.0)
            dur_l       = mse_loss(
                logdur_pred.masked_fill(b["char_mask"], 0.0),
                log_dur_tgt.masked_fill(b["char_mask"], 0.0),
            )

            # Pitch / energy por carácter
            pch_chr = []
            ech_chr = []
            for xi in range(len(b["tokens"])):
                drs  = b["durations"][xi].tolist()
                ptr  = 0
                pc   = []
                ec   = []
                for d in drs:
                    d = max(1, int(round(d)))
                    p = b["pitch"][xi, ptr: ptr + d]
                    e = b["energy"][xi, ptr: ptr + d]
                    pc.append(p.mean() if p.numel() > 0 else torch.tensor(0.0, device=DEVICE))
                    ec.append(e.mean() if e.numel() > 0 else torch.tensor(0.0, device=DEVICE))
                    ptr += d
                pad = b["tokens"].size(1) - len(pc)
                pc += [torch.tensor(0.0, device=DEVICE)] * pad
                ec += [torch.tensor(0.0, device=DEVICE)] * pad
                pch_chr.append(torch.stack(pc[:b["tokens"].size(1)]))
                ech_chr.append(torch.stack(ec[:b["tokens"].size(1)]))

            pch_t = torch.stack(pch_chr)
            ech_t = torch.stack(ech_chr)
            pit_l = mse_loss(
                pitch_pred.masked_fill(b["char_mask"], 0.0),
                pch_t.masked_fill(b["char_mask"], 0.0),
            )
            en_l  = mse_loss(
                energy_pred.masked_fill(b["char_mask"], 0.0),
                ech_t.masked_fill(b["char_mask"], 0.0),
            )

            # Pesos: mel domina, pitch/energy son auxiliares
            loss = mel_l + dur_l + 0.1 * pit_l + 0.1 * en_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses["total"].append(loss.item())
            losses["mel"].append(mel_l.item())
            losses["dur"].append(dur_l.item())
            losses["pitch"].append(pit_l.item())
            losses["energy"].append(en_l.item())

        sched.step()
        final_loss = float(np.mean(losses["total"]))

        if ep % 50 == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - t0
            eta     = elapsed / ep * (epochs - ep)
            now     = datetime.now().strftime("%H:%M:%S")
            lr_now  = opt.param_groups[0]["lr"]
            print(
                f"    [{now}] Época [{ep:04d}/{epochs}] "
                f"loss={final_loss:.4f} "
                f"(mel={np.mean(losses['mel']):.3f} "
                f"dur={np.mean(losses['dur']):.3f} "
                f"pit={np.mean(losses['pitch']):.3f}) "
                f"lr={lr_now:.2e} | ETA {eta/60:.1f}min"
            )

    return final_loss


# ─── Carga de metadata y estadísticas ─────────────────────────────────────────

def _load_metadata(path: str = "dataset_metadata.json") -> List[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception as exc:
        print(f"[!] No se pudo leer {path}: {exc}")
        return []


def _save_stats(mean: float, std: float, path: str = STATS_PATH):
    with open(path, "w") as f:
        json.dump({"mel_mean": mean, "mel_std": std}, f)


def _load_stats(path: str = STATS_PATH) -> Tuple[float, float]:
    if not os.path.exists(path):
        return 0.0, 1.0
    try:
        with open(path) as f:
            d = json.load(f)
        return float(d["mel_mean"]), float(d["mel_std"])
    except Exception:
        return 0.0, 1.0


# ─── Síntesis ─────────────────────────────────────────────────────────────────

def _synthesize(model: AcousticModel,
                vocoder: Optional[NeuralVocoder],
                dsp:     DSP,
                phrase:  str,
                out_wav: str,
                mel_mean: float,
                mel_std:  float) -> str:
    model.eval()
    tokens = torch.LongTensor([text_to_sequence(phrase)]).to(DEVICE)

    with torch.no_grad():
        mel_pred, _, _, _ = model(tokens)

    mel_np = mel_pred.squeeze(0).cpu().numpy()          # [frames, n_mels]
    if mel_np.shape[-1] == N_MELS and mel_np.shape[0] != N_MELS:
        mel_np = mel_np.T                               # → [n_mels, frames]

    # Desnormalizar
    mel_np = mel_np * mel_std + mel_mean

    if vocoder is not None:
        print("[*] Vocoder neuronal...")
        vocoder.eval()
        mel_t  = torch.FloatTensor(mel_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wav = vocoder(mel_t).squeeze(0).cpu().numpy()
        wav = inv_preemphasis(wav)
    else:
        print("[*] Griffin-Lim (fallback)...")
        mag = mel_to_linear(mel_np, dsp.mel_filters)
        wav = griffin_lim(mag, n_iter=60)
        wav = inv_preemphasis(wav)

    wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.95
    write_wav(out_wav, wav, sr=SAMPLE_RATE)
    print(f"[✔] Guardado: {out_wav}  ({len(wav) / SAMPLE_RATE:.2f}s)")
    return out_wav


def _evaluate(model: AcousticModel, dataset: VoiceDataset) -> float:
    model.eval()
    crit  = nn.L1Loss()
    total = 0.0
    n     = 0
    with torch.no_grad():
        for ex in dataset.examples[:50]:
            tok = torch.LongTensor([ex["tokens"]]).to(DEVICE)
            tgt = torch.FloatTensor(ex["mel"]).unsqueeze(0).to(DEVICE)
            pred, _, _, _ = model(tok, target_mel_len=tgt.size(-1))
            total += crit(pred.transpose(1, 2), tgt).item()
            n     += 1
    return total / max(n, 1)


# ─── API pública ──────────────────────────────────────────────────────────────

def train_and_synthesize(
    audio_source:   str,
    output_wav:     str  = "voz_generada.wav",
    test_phrase:    str  = "esta es una síntesis de voz limpia con tonos naturales",
    epochs:         int  = 100,
    batch_size:     int  = 4,
    vocoder_epochs: int  = 60,
    force_retrain:  bool = False,
):
    """
    Pipeline completo:
    1. Carga audio de referencia + metadata con timestamps.
    2. Construye dataset ALINEADO: audio[inicio:fin] ↔ texto real.
    3. Normaliza mels (media/std del dataset).
    4. Entrena modelo acústico FastSpeech 2 por `epochs` épocas.
    5. Entrena vocoder neuronal por `vocoder_epochs` épocas.
    6. Sintetiza la frase de prueba.
    """
    dsp      = DSP()
    metadata = _load_metadata()

    if not metadata:
        raise RuntimeError(
            "No se encontró dataset_metadata.json con timestamps.\n"
            "Ejecuta scraper.py primero para generar el corpus alineado."
        )

    print("\n[*] Cargando audio de referencia...")
    audio = load_audio(audio_source, sr=SAMPLE_RATE)
    print(f"    Audio: {len(audio) / SAMPLE_RATE:.1f}s  |  "
          f"Segmentos en metadata: {len(metadata)}")

    # Primera pasada para calcular estadísticas de mel
    print("[*] Calculando estadísticas del dataset...")
    ds_tmp = VoiceDataset(audio, metadata, dsp, mel_mean=0.0, mel_std=1.0)
    print(f"    Ejemplos construidos (con timestamps): {len(ds_tmp.examples)}")
    if len(ds_tmp.examples) == 0:
        raise RuntimeError("No se generaron ejemplos. Verifica dataset_metadata.json.")

    mel_mean, mel_std = _compute_mel_stats(ds_tmp.examples)
    _save_stats(mel_mean, mel_std)
    print(f"    Mel media: {mel_mean:.3f}  |  std: {mel_std:.3f}")

    # Dataset final normalizado
    dataset = VoiceDataset(audio, metadata, dsp, mel_mean=mel_mean, mel_std=mel_std)

    model   = AcousticModel().to(DEVICE)
    vocoder: Optional[NeuralVocoder] = None

    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"[*] Cargando modelo acústico desde '{MODEL_PATH}'...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[✔] Modelo cargado.")
        except RuntimeError as e:
            print(f"[!] Modelo incompatible (arquitectura cambió): {str(e)[:120]}...")
            print("[!] Eliminando modelo viejo y reentrenando desde cero.")
            os.remove(MODEL_PATH)
            if os.path.exists(VOCODER_PATH):
                os.remove(VOCODER_PATH)
            final_loss = _train_model(model, dataset, epochs=epochs, batch_size=batch_size)
            print(f"\n[✔] Entrenamiento finalizado. Loss final: {final_loss:.6f}")
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[✔] Modelo guardado en '{MODEL_PATH}'")
            val_l1 = _evaluate(model, dataset)
            print(f"    L1 mel (validación interna): {val_l1:.6f}")
    else:
        final_loss = _train_model(model, dataset, epochs=epochs, batch_size=batch_size)
        print(f"\n[✔] Entrenamiento finalizado. Loss final: {final_loss:.6f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[✔] Modelo guardado en '{MODEL_PATH}'")
        val_l1 = _evaluate(model, dataset)
        print(f"    L1 mel (validación interna): {val_l1:.6f}")

    if os.path.exists(VOCODER_PATH) and not force_retrain:
        print(f"[*] Cargando vocoder desde '{VOCODER_PATH}'...")
        try:
            vocoder = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
            vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=DEVICE))
            print("[✔] Vocoder cargado.")
        except RuntimeError:
            print("[!] Vocoder incompatible. Reentrenando...")
            os.remove(VOCODER_PATH)
            vocoder = train_neural_vocoder(audio, dsp, epochs=vocoder_epochs)
            torch.save(vocoder.state_dict(), VOCODER_PATH)
    else:
        vocoder = train_neural_vocoder(audio, dsp, epochs=vocoder_epochs)
        torch.save(vocoder.state_dict(), VOCODER_PATH)
        print(f"[✔] Vocoder guardado en '{VOCODER_PATH}'")

    print(f"\n[*] Sintetizando: '{test_phrase}'")
    _synthesize(model, vocoder, dsp, test_phrase, output_wav, mel_mean, mel_std)


def synthesize_only(
    phrase:       str,
    output_wav:   str = "voz_generada.wav",
    audio_source: str = "audio_referencia.wav",
):
    """Sintetiza una frase usando modelos ya entrenados."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró '{MODEL_PATH}'. Ejecuta train_and_synthesize() primero."
        )
    dsp              = DSP()
    mel_mean, mel_std = _load_stats()
    model = AcousticModel().to(DEVICE)
    print(f"[*] Cargando modelo desde '{MODEL_PATH}'...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError:
        raise RuntimeError(
            f"El modelo en '{MODEL_PATH}' es incompatible con la arquitectura actual. "
            "Ejecuta train_and_synthesize(force_retrain=True) para reentrenar."
        )

    vocoder: Optional[NeuralVocoder] = None
    if os.path.exists(VOCODER_PATH):
        try:
            vocoder = NeuralVocoder(n_mels=N_MELS, hop_length=HOP_LENGTH).to(DEVICE)
            vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=DEVICE))
        except RuntimeError:
            print("[!] Vocoder incompatible, se usará Griffin-Lim como fallback.")
            vocoder = None

    print(f"[*] Sintetizando: '{phrase}'")
    _synthesize(model, vocoder, dsp, phrase, output_wav, mel_mean, mel_std)


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    AUDIO  = ("audio_referencia.wav" if os.path.exists("audio_referencia.wav")
              else "https://youtu.be/bhQnooudZcs?si=WLtSJNfCwZ1EZyQD")
    FRASE  = "esta es una síntesis de voz limpia con tonos naturales corriendo en mi procesador M4"
    train_and_synthesize(AUDIO, test_phrase=FRASE, epochs=100, vocoder_epochs=60)
