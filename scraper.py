import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_SAMPLE_RATE = 22050
DEFAULT_OUTPUT_AUDIO = "audio_referencia.wav"
DEFAULT_OUTPUT_METADATA = "dataset_metadata.json"
DEFAULT_SUBTITLES_BASE = "subtitulos"


def _which(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def _run(cmd: List[str], cwd: Optional[str] = None, capture: bool = True) -> subprocess.CompletedProcess:
    kwargs = {"cwd": cwd}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def descargar_audio_alta_calidad(
    url_video: str,
    output_wav: str = DEFAULT_OUTPUT_AUDIO,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Optional[str]:
    """
    Descarga el stream de audio de mayor calidad disponible y lo convierte a
    WAV mono PCM 16-bit al sample rate deseado (por defecto 22050 Hz).

    Requiere yt-dlp y ffmpeg instalados en el PATH.
    """
    if not _which("yt-dlp"):
        raise RuntimeError("'yt-dlp' no está en el PATH.")
    if not _which("ffmpeg"):
        raise RuntimeError("'ffmpeg' no está en el PATH.")

    output_path = Path(output_wav).resolve()
    tmp_wav = output_path.with_suffix(".tmp.wav")

    print(f"[*] Descargando audio de alta calidad desde: {url_video}")
    print(f"    Formato objetivo: WAV mono {sample_rate} Hz / 16-bit -> {output_path.name}")

    try:
        # yt-dlp: elige el mejor stream de audio puro y lo envía por stdout.
        # '-f bestaudio' prioriza calidad; '--audio-quality 0' fuerza la mejor
        # calidad al re-encodear, aunque aquí solo pedimos el stream original.
        ytdlp_cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--audio-quality", "0",
            "--no-playlist",
            "-o", "-",
            url_video,
        ]

        # ffmpeg escribe directamente un WAV temporal con cabecera RIFF.
        # El formato se infiere de la extensión .wav, evitando el problema
        # de output format con archivos .raw.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", "pipe:0",
            "-vn",                     # sin video
            "-acodec", "pcm_s16le",    # PCM 16-bit little-endian
            "-ar", str(sample_rate),   # sample rate objetivo
            "-ac", "1",                # mono
            "-af", "highpass=f=60,lowpass=f=8000,loudnorm",  # filtro básico + normalización
            str(tmp_wav),
        ]

        p1 = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1.stdout.close()

        _, stderr_ytdlp = p1.communicate()
        _, stderr_ffmpeg = p2.communicate()

        if p2.returncode != 0:
            print("[!] ffmpeg falló al convertir el audio.")
            if stderr_ffmpeg:
                print(stderr_ffmpeg.decode("utf-8", errors="replace")[-800:])
            return None
        if p1.returncode != 0:
            print("[!] yt-dlp falló al descargar el audio.")
            if stderr_ytdlp:
                print(stderr_ytdlp.decode("utf-8", errors="replace")[-800:])
            return None

        if not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
            print("[!] No se generó archivo de audio.")
            return None

        tmp_wav.replace(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[✔] Audio guardado: {output_path} ({size_mb:.2f} MiB)")
        return str(output_path)

    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()


def descargar_subtitulos(
    url_video: str,
    output_base: str = DEFAULT_SUBTITLES_BASE,
) -> Optional[str]:
    """
    Descarga subtítulos automáticos de YouTube en formato json3.
    Retorna la ruta al archivo json3 descargado o None si falla.
    """
    print("[*] Extrayendo subtítulos automáticos de YouTube para alineación...")

    comando_subs = [
        "yt-dlp",
        "--write-auto-subs",
        "--skip-download",
        "--sub-format", "json3",
        "--sub-lang", "es",
        "--no-playlist",
        "-o", output_base,
        url_video,
    ]
    res = _run(comando_subs)

    archivo_json = f"{output_base}.es.json3"
    if not os.path.exists(archivo_json):
        archivos = [f for f in os.listdir(".") if f.endswith(".json3")]
        if archivos:
            archivo_json = archivos[0]
        else:
            print("[!] No se pudieron descargar los subtítulos automáticos.")
            return None

    print(f"[✔] Subtítulos guardados: {archivo_json}")
    return archivo_json


def limpiar_texto_subtitulo(texto: str) -> str:
    """Limpia caracteres no deseados y convierte a minúsculas."""
    texto = texto.lower().strip()
    permitidos = " abcdefghijklmnopqrstuvwxyzñáéíóúüü,.!?-"
    return "".join(c for c in texto if c in permitidos)


def extraer_metadata_alineada(
    archivo_json: str,
    max_frases: int = 200,
    min_longitud: int = 5,
    max_longitud: int = 120,
) -> List[dict]:
    """
    Procesa un archivo json3 de subtítulos y retorna frases alineadas
    con inicio/fin en segundos.
    """
    with open(archivo_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_alineado = []
    print("[*] Procesando bloques de texto y marcas de tiempo...")

    for event in data.get("events", []):
        if "segs" not in event or "tStartMs" not in event:
            continue

        texto = "".join(seg.get("utf8", "") for seg in event["segs"])
        texto = limpiar_texto_subtitulo(texto)

        if not (min_longitud <= len(texto) <= max_longitud):
            continue

        inicio_seg = event["tStartMs"] / 1000.0
        duracion_seg = event.get("dDurationMs", 0) / 1000.0

        dataset_alineado.append({
            "texto": texto,
            "inicio": inicio_seg,
            "fin": inicio_seg + duracion_seg,
        })

    dataset_alineado = dataset_alineado[:max_frases]
    return dataset_alineado


def guardar_metadata(
    dataset: List[dict],
    output_path: str = DEFAULT_OUTPUT_METADATA,
) -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"[✔] Metadata alineada guardada: {output_path} ({len(dataset)} frases)")
    return output_path


def preparar_corpus_voz(
    url_video: str,
    output_wav: str = DEFAULT_OUTPUT_AUDIO,
    output_metadata: str = DEFAULT_OUTPUT_METADATA,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """
    Pipeline completo: descarga audio de alta calidad + subtítulos alineados.
    Los archivos generados están pensados para ser consumidos por voice.py.
    """
    audio_path = descargar_audio_alta_calidad(url_video, output_wav, sample_rate)
    if audio_path is None:
        print("[X] Falló la descarga de audio. Abortando.")
        sys.exit(1)

    archivo_json = descargar_subtitulos(url_video)
    if archivo_json:
        dataset = extraer_metadata_alineada(archivo_json)
        guardar_metadata(dataset, output_metadata)
    else:
        print("[!] Continuando sin metadata de subtítulos.")

    print("\n[✔] Corpus de voz listo.")
    print(f"    Audio: {audio_path}")
    print(f"    Metadata: {output_metadata}")


if __name__ == "__main__":
    url = "https://youtu.be/bhQnooudZcs?si=WLtSJNfCwZ1EZyQD"
    preparar_corpus_voz(url)
