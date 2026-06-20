# Voice AI YouTube

**Autor:** Pablo Aravena  
**Licencia:** MIT

Este proyecto enseña a una computadora a **imitar una voz humana** usando como referencia videos largos de YouTube. No usa servicios de voz listos para usar (como los de grandes empresas): construye el sistema desde cero con matemática, señales de audio y redes neuronales.

El flujo tiene dos partes:

1. **`scraper.py`** — descarga el audio y los subtítulos del video.
2. **`voice.py`** — entrena un modelo y genera frases nuevas con esa voz.

Repositorio: [github.com/Bjunk/voice_ai_youtube](https://github.com/Bjunk/voice_ai_youtube)

---

## Como en *Misión Imposible*

Si viste la saga de películas, recordarás una escena donde **Ethan Hunt** necesita imitar la voz de un maleante internacional. Para lograrlo, le pide que **recite un poema** — o que hable un rato — mientras un sistema de inteligencia artificial **escucha y aprende** cómo suena: el tono, el ritmo, las pausas, la forma de pronunciar.

Después, con esas muestras, la IA puede **generar frases nuevas** con la voz de esa persona, como si él mismo las hubiera dicho.

**Este proyecto hace algo parecido, pero con un video de YouTube en lugar de un poema en una habitación:**

| En la película | En este proyecto |
|----------------|------------------|
| El maleante recita un poema bajo vigilancia | Un video largo de YouTube con horas de habla |
| El sistema graba y analiza la voz | `scraper.py` descarga el audio y los subtítulos |
| La IA aprende timbre, ritmo y entonación | `voice.py` entrena el modelo con espectrogramas, pitch y energía |
| Replican la voz para decir lo que quieran | El modelo sintetiza frases nuevas en `voz_generada.wav` |

La diferencia es que aquí no buscamos perfección de espía de Hollywood: entrenamos desde cero con herramientas abiertas, y la calidad depende del video, los subtítulos y el tiempo de entrenamiento. Pero la **lógica es la misma**: escuchar a alguien hablar lo suficiente para clonar su forma de hablar.

---

## ¿Qué problema resuelve?

La voz humana no es solo “leer texto en voz alta”. Cada persona tiene:

- un **timbre** (cómo suena su garganta y resonancia),
- un **ritmo** (pausas, velocidad),
- una **entonación** (subidas y bajadas de tono),
- y un **estilo** (formal, relajado, técnico, etc.).

Los sistemas comerciales de voz suelen depender de modelos gigantes ya entrenados. Aquí la idea es distinta: **tomar un video de YouTube de muchos minutos**, extraer la voz de quien habla, y entrenar un modelo propio que aprenda ese estilo.

---

## Base científica (explicada simple)

### 1. El sonido como imagen

El audio se convierte en un **espectrograma Mel**: una “foto” del sonido donde el eje horizontal es el tiempo y el vertical son las frecuencias que el oído humano percibe mejor. Es la misma idea que usan sistemas modernos de voz como FastSpeech 2.

### 2. Separar “qué se dice” de “cómo suena”

El entrenamiento divide el problema en dos capas:

| Capa | Qué hace | Analogía |
|------|----------|----------|
| **Modelo acústico** | Recibe texto y predice el espectrograma (forma de la voz) | El “cerebro” que decide cómo hablar |
| **Vocoder** | Convierte el espectrograma en audio audible | La “boca” que produce el sonido |

### 3. Alinear texto con audio (DTW)

Para aprender, el modelo necesita saber **qué parte del audio corresponde a qué palabra**. Eso se llama *alineación texto-audio*.

En este proyecto se usa **Dynamic Time Warping (DTW)**: un algoritmo que estira o comprime secuencias para encontrar la mejor correspondencia entre caracteres del texto y frames del espectrograma, aunque no tengan la misma duración.

### 4. Prosodia: tono y energía

Además del espectrograma, el modelo aprende:

- **F0 (pitch)**: frecuencia fundamental de la voz (tono alto o bajo), estimada con autocorrelación tipo YIN.
- **Energía**: intensidad de cada frame (volumen relativo).

Estos dos valores ayudan a capturar la **expresividad**: énfasis, pausas, entonación de preguntas, etc.

### 5. Arquitectura FastSpeech 2 (simplificada)

El modelo acústico está inspirado en **FastSpeech 2**, un diseño no autoregresivo (genera todo el audio de una vez, sin ir palabra por palabra como Tacotron). Incluye:

- codificador de caracteres con bloques FFT (convoluciones),
- predictores de duración, pitch y energía,
- regulador de longitud (expande cada carácter según cuánto dura al hablar),
- decodificador que produce el espectrograma final.

### 6. Vocoder neuronal (en lugar de Griffin-Lim)

Los primeros sistemas usaban **Griffin-Lim** para reconstruir el audio desde el espectrograma, pero suena metálico y artificial.

Este proyecto entrena un **vocoder neuronal ligero** (inspirado en MelGAN/HiFi-GAN) que aprende directamente a convertir espectrogramas en ondas de sonido, con mejor calidad que métodos clásicos.

---

## ¿Por qué videos largos de YouTube?

Un video de una hora puede parecer demasiado, pero aporta ventajas:

- **Más ejemplos de voz**: el audio se parte en segmentos de ~6 segundos; un video de 60 minutos genera cientos de fragmentos.
- **Variedad de entonación**: preguntas, énfasis, pausas, cambios de ritmo.
- **Estilo del hablante**: muletillas, velocidad, tono emocional.

`scraper.py` descarga:

- el **mejor stream de audio** disponible (`bestaudio` con yt-dlp),
- lo convierte a WAV mono 22 050 Hz con filtros de limpieza,
- y extrae **subtítulos automáticos en español** con marcas de tiempo exactas.

Esos subtítulos alimentan `dataset_metadata.json`: frases con `inicio` y `fin` en segundos, que `voice.py` usa como corpus de entrenamiento.

---

## ¿Cómo replica el estilo del lenguaje?

El modelo no “copia” frases del video. Aprende patrones:

1. **Timbre**: codificado en los espectrogramas Mel de cada segmento.
2. **Ritmo**: el predictor de duración aprende cuánto tiempo dura cada carácter al hablar.
3. **Entonación**: pitch y energía condicionan la generación.
4. **Vocabulario y forma de hablar**: las frases de los subtítulos enseñan el estilo textual del hablante.

Al sintetizar una frase nueva, el modelo combina texto + duraciones + pitch + energía para producir un espectrograma con el “sello” de la voz aprendida, y el vocoder lo convierte en audio.

---

## Lo innovador de este enfoque

| Aspecto tradicional | Este proyecto |
|--------------------|---------------|
| Datasets profesionales (LJSpeech, horas grabadas en estudio) | Video de YouTube cualquiera con subtítulos |
| Librerías TTS pre-entrenadas (Coqui, MeloTTS, etc.) | Implementación propia con PyTorch |
| Vocoder comercial (HiFi-GAN pre-entrenado) | Vocoder neuronal entrenado con el mismo audio |
| Alineación forzada (MFA, Montreal Forced Aligner) | DTW implementado desde cero |
| Fonemas profesionales | Caracteres + subtítulos automáticos |

La propuesta es **democratizar** el entrenamiento de voz: solo necesitas un link de YouTube, Python, yt-dlp y ffmpeg.

---

## Dificultades de entrenar desde cero

Entrenar un sistema de voz sin modelos pre-entrenados es **mucho más difícil** que usar uno comercial. Estas son las principales barreras:

### Datos imperfectos
- Los subtítulos automáticos de YouTube tienen errores, retrasos y frases cortadas.
- No hay transcripción fonética precisa (fonemas).
- La alineación DTW es aproximada, no exacta.

### Recursos computacionales
- Un video de ~60 minutos genera cientos de ejemplos y el entrenamiento puede tardar **horas** (300 épocas del modelo acústico + 50 del vocoder).
- Se recomienda GPU o aceleración MPS (Apple Silicon).

### Calidad del vocoder
- Griffin-Lim suena robótico; el vocoder neuronal mejora, pero no alcanza HiFi-GAN pre-entrenado en LJSpeech.
- Entrenar un vocoder de calidad profesional requiere decenas de miles de pasos adicionales.

### Generalización limitada
- El modelo aprende la voz del video, no un hablante genérico.
- Frases muy distintas al estilo del video pueden sonar extrañas.
- No hay control fino de emociones ni múltiples hablantes.

### Épocas de entrenamiento
- Con pocas épocas (30–50) el audio es inteligible pero poco natural.
- Referencias académicas de FastSpeech 2 usan **160 000 epoch** en datasets limpios de 24 horas.
- Para este proyecto, **100–300 épocas** son un punto de partida razonable, monitoreando la pérdida (loss).

---

## Otras técnicas posibles (comparación)

| Técnica | Ventaja | Desventaja |
|---------|---------|------------|
| **Tacotron 2** (autoregresivo) | Muy natural en datasets limpios | Lento en inferencia, inestable |
| **FastSpeech 2** (este proyecto) | Rápido, paralelo, estable | Necesita alineación previa |
| **VITS** (end-to-end) | Alta calidad, un solo modelo | Complejo, muchos datos |
| **XTTS / Coqui TTS** (pre-entrenado) | Clonación con pocos segundos | Dependencia externa, no educativo |
| **Griffin-Lim** (vocoder clásico) | Sin entrenamiento | Calidad baja, sonido metálico |
| **HiFi-GAN pre-entrenado** | Excelente calidad | Requiere torchaudio y mels compatibles |
| **Montreal Forced Aligner** | Alineación precisa | Instalación compleja |
| **Whisper (transcripción)** | Subtítulos más exactos que YouTube auto | Más lento, otro modelo |

Este proyecto eligió FastSpeech 2 + vocoder neuronal propio + DTW + subtítulos de YouTube como **balance entre aprendizaje, control y simplicidad**.

---

## Requisitos

- Python 3.10+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)
- Paquetes Python: `numpy`, `torch`, `scipy` (opcional, mejora DTW)

```bash
pip install numpy torch scipy
```

---

## Uso rápido

### Paso 1: Preparar el corpus desde YouTube

```bash
python3 scraper.py
```

Genera:
- `audio_referencia.wav` — audio limpio mono 22 050 Hz
- `dataset_metadata.json` — frases con timestamps
- `subtitulos.es.json3` — subtítulos crudos

### Paso 2: Entrenar y sintetizar

```bash
python3 voice.py
```

Entrena el modelo acústico y el vocoder (si no existen), guarda:
- `voice_model.pt`
- `voice_vocoder.pt`

Y genera `voz_generada.wav` con una frase de prueba.

### Paso 3: Probar otra frase (sin reentrenar)

```python
from voice import synthesize_only

synthesize_only("hola, esta es una prueba de mi voz clonada", "prueba.wav")
```

---

## Estructura del proyecto

```
voice_ai_youtube/
├── scraper.py              # Descarga audio + subtítulos de YouTube
├── voice.py                # Entrenamiento TTS + síntesis
├── dataset_metadata.json   # Frases alineadas (generado)
├── audio_referencia.wav    # Audio de referencia (generado, no en git)
├── voice_model.pt          # Modelo acústico entrenado (generado)
├── voice_vocoder.pt        # Vocoder neuronal (generado)
└── LICENSE
```

---

## Flujo del sistema

```
YouTube (video largo)
        │
        ▼
   scraper.py
   ├── audio_referencia.wav
   └── dataset_metadata.json
        │
        ▼
    voice.py
   ├── Espectrogramas Mel + pitch + energía
   ├── Alineación DTW (texto ↔ audio)
   ├── Entrenamiento FastSpeech 2
   ├── Entrenamiento vocoder neuronal
   └── voz_generada.wav
```

---

## Créditos y referencias

### Ciencia y tecnología
- **FastSpeech 2**: Ren et al., 2020 — [arXiv:2006.04558](https://arxiv.org/abs/2006.04558)
- **Griffin-Lim**: reconstrucción de fase iterativa
- **YIN**: estimación de pitch por autocorrelación
- **DTW**: Dynamic Time Warping para alineación de secuencias
- **HiFi-GAN / MelGAN**: inspiración para el vocoder neuronal

### Inspiración cultural
- ***Mission: Impossible – Dead Reckoning Part One*** (2023, dir. Christopher McQuarrie) — escena en la que Ethan Hunt obtiene muestras de la voz de un objetivo (incluido hacerlo hablar/recitar texto) para que un sistema de IA aprenda su timbre, ritmo y entonación y pueda replicarla con frases nuevas. Esa idea de “escuchar primero, clonar después” es la metáfora central de este proyecto.  
  - [IMDb](https://www.imdb.com/title/tt9603212/) · [Paramount Pictures](https://www.paramountmovies.com/movies/mission-impossible-dead-reckoning-part-one)
- **Saga *Mission: Impossible*** (Paramount / Christopher McQuarrie) — tradición del IMF de suplantar identidades; en entregas recientes la clonación de voz por IA reemplaza o complementa las máscaras clásicas de las películas anteriores.

---

## Autor

**Pablo Aravena**  
Proyecto de investigación de los domingos aburridos y experimentación en síntesis de voz a partir de contenido de YouTube.

Si usas este código, menciona al autor y respeta la licencia MIT.
