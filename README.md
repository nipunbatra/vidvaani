# yt-hindi

AI-powered YouTube to Hindi dubbing pipeline. Automatically transcribes, translates, and dubs YouTube videos to Hindi with natural-sounding voices.

## Features

- **Auto Intro Detection**: Automatically detects intro music and preserves it
- **Multiple TTS Backends**: Sarvam AI (native Indian voices), Gemini TTS, Edge TTS (free)
- **Non-Speech Preservation**: Keeps original audio during pauses for natural sound
- **SRT Subtitles**: Generates Hindi subtitle files, can burn into video
- **Parallel Processing**: TTS runs in parallel, results are cached
- **Cost Tracking**: Real-time API cost display

## Installation

```bash
# Clone the repository
git clone https://github.com/nipunbatra/yt-hindi.git
cd yt-hindi

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Requirements

- Python 3.11+
- ffmpeg
- API keys (set as environment variables):
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` - for translation and Gemini TTS
  - `SARVAM_API_KEY` - for Sarvam AI TTS (optional)

## Quick Start

```bash
# Dub a YouTube video (demo mode - first 5 segments)
yt-hindi dub "https://www.youtube.com/watch?v=VIDEO_ID"

# Full video with Sarvam voice
yt-hindi dub "https://www.youtube.com/watch?v=VIDEO_ID" --full -b sarvam -v abhilash

# With video title as folder name
yt-hindi dub "https://www.youtube.com/watch?v=VIDEO_ID" --full --title-folder
```

## Usage

### Dub Command

```bash
yt-hindi dub URL [OPTIONS]
```

**Options:**
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
yt-hindi burn-subs video.mp4 subtitles.srt --position bottom_bar

# Overlay on video
yt-hindi burn-subs video.mp4 subtitles.srt --position overlay
```

### Generate Voice Samples

```bash
yt-hindi samples -o ./samples
```

## TTS Voices

### Sarvam AI (Native Indian)
| Voice | Gender |
|-------|--------|
| abhilash | Male |
| karun | Male |
| hitesh | Male |
| vidya | Female |
| anushka | Female |
| manisha | Female |
| arya | Female |

### Gemini TTS
| Voice | Gender |
|-------|--------|
| Charon | Male |
| Fenrir | Male |
| Aoede | Female |
| Kore | Female |
| Puck | Male |

### Edge TTS (Free)
| Voice | Gender |
|-------|--------|
| male | Madhur |
| female | Swara |

## Pipeline

```
YouTube URL
    ↓
1. Download video (yt-dlp)
    ↓
2. Auto-detect intro music (ffmpeg silencedetect)
    ↓
3. Transcribe audio (MLX Whisper)
    ↓
4. Translate to Hindi (Gemini 2.0 Flash)
    ↓
5. Generate Hindi TTS (Sarvam/Gemini/Edge)
    ↓
6. Assemble final video (ffmpeg)
    ↓
Hindi dubbed video + SRT subtitles
```

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

## Cost Estimates

| Video Length | Translation | Sarvam TTS | Gemini TTS | Total |
|--------------|-------------|------------|------------|-------|
| 5 min | ~$0.001 | ~$0.05 | ~$0.02 | ~$0.05 |
| 20 min | ~$0.002 | ~$0.20 | ~$0.08 | ~$0.20 |

## Demo

Open `demo.html` in a browser to see example dubbed videos.

## Examples

```bash
# NPTEL Deep Learning lecture with Sarvam voice
yt-hindi dub "https://www.youtube.com/watch?v=4TC5s_xNKSs" \
    --full --title-folder -b sarvam -v abhilash

# Generate multiple voices (translation is cached)
yt-hindi dub "https://www.youtube.com/watch?v=4TC5s_xNKSs" \
    --full --title-folder -b sarvam -v karun
yt-hindi dub "https://www.youtube.com/watch?v=4TC5s_xNKSs" \
    --full --title-folder -b sarvam -v hitesh

# Burn subtitles into video
yt-hindi burn-subs output/Video_Title/video_hindi_abhilash.mp4 \
    output/Video_Title/video_hindi.srt --position bottom_bar
```

## License

MIT

## Author

[Nipun Batra](https://nipunbatra.github.io) - IIT Gandhinagar
