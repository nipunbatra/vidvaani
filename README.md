# VidVaani

AI-powered pipeline that dubs YouTube videos into Hindi. It downloads a video, transcribes it locally, translates the transcript, synthesizes natural Hindi speech, and reassembles the video — preserving intro music, pauses, and background audio.

**Demo page:** https://nipunbatra.github.io/vidvaani/ · **Slides:** [HTML](https://nipunbatra.github.io/vidvaani/slides/vidvaani.html) / [PDF](https://nipunbatra.github.io/vidvaani/slides/vidvaani.pdf)

## Features

- **Auto intro detection** — detects intro music and preserves it in the dub
- **Multiple TTS backends** — Sarvam AI (native Indian voices), Gemini TTS, Edge TTS (free)
- **Non-speech preservation** — keeps original audio during pauses for natural sound
- **SRT subtitles** — generates Hindi subtitle files, optionally burned into the video
- **Parallel TTS with caching** — resume failed runs; re-dub with a new voice without re-translating
- **Cost tracking** — real-time API cost display per run

## Pipeline

```
YouTube URL
    |
 1. Download video ............ yt-dlp
 2. Detect intro music ........ ffmpeg silencedetect
 3. Transcribe (local) ........ MLX Whisper (distil-large-v3)
 4. Translate to Hindi ........ Gemini Flash
 5. Synthesize Hindi speech ... Sarvam AI / Gemini TTS / Edge TTS
 6. Reassemble video .......... ffmpeg (duration matching, non-speech preservation)
    |
Hindi dubbed video + SRT subtitles
```

## Installation

```bash
git clone https://github.com/nipunbatra/vidvaani.git
cd vidvaani

# Install with uv
uv sync
```

## Requirements

- Python 3.11+, ffmpeg
- Apple Silicon Mac for local MLX Whisper transcription
- API keys (environment variables):
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` — translation and Gemini TTS
  - `SARVAM_API_KEY` — Sarvam AI TTS (optional)

## Quick Start

```bash
# Dub a YouTube video (demo mode - first 5 segments)
uv run vidvaani dub "https://www.youtube.com/watch?v=VIDEO_ID"

# Full video with a Sarvam voice
uv run vidvaani dub "https://www.youtube.com/watch?v=VIDEO_ID" --full -b sarvam -v abhilash

# With video title as folder name
uv run vidvaani dub "https://www.youtube.com/watch?v=VIDEO_ID" --full --title-folder
```

## Usage

### Dub Command

```bash
vidvaani dub URL [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `./output` | Output directory |
| `-v, --voice` | `Kore` | Voice name (see below) |
| `-b, --backend` | `gemini` | TTS backend: `gemini`, `sarvam`, `edge` |
| `--full` | - | Process all segments (not just first 5) |
| `-n, --segments` | `5` | Number of segments to process |
| `--title-folder` | - | Create folder based on video title |
| `--intro-offset` | auto | Seconds to skip (auto-detects intro music) |
| `--keep-original` | - | Keep original audio at low volume |
| `--no-preserve-music` | - | Don't preserve non-speech audio |

### Burn Subtitles

```bash
# Burn Hindi subtitles into video (black bar below)
vidvaani burn-subs video.mp4 subtitles.srt --position bottom_bar

# Overlay on video
vidvaani burn-subs video.mp4 subtitles.srt --position overlay
```

### Generate Voice Samples

```bash
vidvaani samples -o ./samples
```

## TTS Voices

| Backend | Male | Female | Notes |
|---------|------|--------|-------|
| Sarvam AI (`sarvam`) | abhilash, karun, hitesh | vidya, anushka, manisha, arya | Native Indian prosody |
| Gemini TTS (`gemini`) | Charon, Orus, Iapetus, Sadaltager, Fenrir, Puck | Kore, Aoede | High quality |
| Edge TTS (`edge`) | male (Madhur) | female (Swara) | Free |

## Output Files

```
output/
└── Video_Title/
    ├── VIDEO_ID.mp4              # Original video
    ├── VIDEO_ID_hindi_voice.mp4  # Hindi dubbed video
    ├── VIDEO_ID_hindi.srt        # Hindi subtitles
    ├── VIDEO_ID_transcript_en.json
    ├── VIDEO_ID_transcript_hi.json
    └── tts_segments_voice/       # Cached TTS audio
```

## Cost and Time

The pipeline prints an exact cost and timing breakdown after every run, computed
from API-reported token counts at official prices. Measured on a 7-minute NPTEL
lecture (July 2026, Apple Silicon):

| Backend | Cost (7 min) | Extrapolated, 1 hour | End-to-end time |
|---------|-------------|----------------------|-----------------|
| Sarvam bulbul:v2 | Rs 7-10 | ~Rs 80 | 2.5-4 min |
| Gemini 2.5 Flash TTS | Rs 11 | ~Rs 95 | similar |
| Edge TTS (free) | Rs 1.4 (translation only) | ~Rs 12 | similar |

Timing is dominated by the translation API; transcription is local. Re-runs
with a different voice reuse cached transcripts and translations.

## Fully local (experimental)

The two cloud stages have working open-weights replacements, run entirely
on-device via MLX: **Gemma 4** for translation and **Qwen3-TTS 1.7B** for
speech, including cross-lingual voice cloning (an English reference of the
lecturer speaking Hindi in their own voice). Measured on a 58 s clip
(July 2026, Apple Silicon): translation 59 s with Gemma 4 31B 4-bit, cloned
speech at 1.6× real-time, ~3 min end-to-end, Rs 0.

Voice-clone fidelity was then pushed in measured rungs — single take 0.76 →
scored best-of-N search 0.85 → 7-minute laptop LoRA 0.86 → full SFT on a lab
A100 trained on a 175-minute speaker-verified dataset → **0.89**, against a
0.93 real-voice self-similarity ceiling (ECAPA cosine; other voices ~0.3),
with an STT content check gating every published take. See the "Fully local"
and "Your own voice" cards on the demo page and the full method + numbers in
[docs/local-models.md](docs/local-models.md). Cloned voices are published
only with the speaker's consent.

## Design notes

How the pipeline fits Hindi speech into the original timing — and how that
design compares to the automatic-dubbing literature — is documented in
[docs/timing-alignment.md](docs/timing-alignment.md).

## Examples

```bash
# NPTEL Deep Learning lecture with Sarvam voice
vidvaani dub "https://www.youtube.com/watch?v=4TC5s_xNKSs" \
    --full --title-folder -b sarvam -v abhilash

# Generate multiple voices (translation is cached)
vidvaani dub "https://www.youtube.com/watch?v=4TC5s_xNKSs" \
    --full --title-folder -b sarvam -v karun

# Burn subtitles into video
vidvaani burn-subs output/Video_Title/video_hindi_abhilash.mp4 \
    output/Video_Title/video_hindi.srt --position bottom_bar
```

## License

MIT

## Author

[Nipun Batra](https://nipunbatra.github.io) — IIT Gandhinagar
