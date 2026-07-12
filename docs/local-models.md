# Going fully local: open-weights models for the pipeline

Survey (July 2026) plus measured experiments on Apple Silicon. Goal: replace
the two cloud stages (translation, TTS) with local open-weights models.

## Measured results, second run (58 s probability clip, M-series Mac, 12 Jul 2026)

Published as the "Your own voice" card on the demo page: the author's own
probability clip (joint-PDF question in GeoGebra, from the mini-demo gallery),
translated by **Gemma 4 31B** and spoken in **his own cloned voice** by
Qwen3-TTS — MLX end to end, no ollama.

| Stage | Model | Time | Notes |
|---|---|---|---|
| Transcribe | MLX Whisper distil-large-v3 | 23.4 s (5 segments) | Includes one-time model fetch; warm ≈ 4 s |
| Translate | `mlx-community/gemma-4-31b-it-4bit` via `mlx-vlm` | 59.2 s (5 segments) | Warm model load 6.8 s (first load added a one-time 17 GB download). Followed word budgets; unlike the 12B run it did NOT fix Whisper mishearings ("Geozevra") |
| TTS (cloned) | Qwen3-TTS 1.7B Base (`mlx-audio` 0.3.1) + 28.5 s English reference from the clip itself | 95.3 s for 58.3 s audio, **RTF 1.63** | Much faster than the July RTF ~4 (newer mlx-audio); model load 5.4 s. Gemini STT round-trip word-perfect; it even pronounced the misheard "Geozevra" close enough that STT heard "GeoGebra" |
| Assemble | normal pipeline (trim, atempo fit, mix, loudnorm) | 9.0 s | Dub loudness −24.1 LUFS = source −24.1 LUFS |

End-to-end warm compute ≈ 187 s for the 58.5 s clip (~3.2× real-time, ₹0).
At RTF 1.6 a 1-hour lecture is now a ~2–2.5 h batch, not an overnight one.

## Measured results, first run (58 s demo clip, M-series Mac, 11 Jul 2026)

| Stage | Model | Time | Notes |
|---|---|---|---|
| Translate | Gemma 4 12B via ollama | 24.5 s (5 segments) | Quality close to Gemini 2.5 Flash; needed one extra prompt rule ("spoken words only — no LaTeX/parentheses"); even corrected a Whisper mishearing |
| Translate | Gemma 3 12B via ollama | 26.4 s | Slightly weaker phrasing, still usable |
| TTS (default voice) | Qwen3-TTS 1.7B (MLX, `mlx-audio`) | RTF ≈ 4.4 | Hindi is NOT in its official language list but works empirically (verified by Sarvam STT round-trip) |
| TTS (cloned voice) | Qwen3-TTS + 30 s English reference | RTF ≈ 4.0 | Cross-lingual cloning: English reference → same voice speaking Hindi, word-perfect STT round-trip |
| Model load | Qwen3-TTS | 21.8 s once | |

The "Fully local" card on the demo page = local Whisper + Gemma 4 12B +
Qwen3-TTS (default voice), assembled by the normal pipeline. ₹0 per lecture.

**Voice cloning & consent:** the first cloned-voice output (11 Jul) came from
a lecture by a third-party professor and was therefore NOT published. The
published "Your own voice" card (12 Jul) clones the author's own voice from
his own clip — exactly the compelling legitimate use: a lecturer dubbing his
own course in his own voice. Publish cloned voices only with the speaker's
consent.

## TTS landscape for Hindi + voice cloning (researched July 2026)

Ranked for "clone a lecturer from ~30 s of English audio → natural Hindi, on a Mac":

1. **Chatterbox Multilingual-hi** (Resemble AI) — MIT, Hindi pack, cloning from
   ~10 s reference *without* transcript, cross-lingual documented. PyTorch MPS
   RTF ~4.6; the community MLX conversion currently fails to load in
   mlx-audio (weight-key mismatch). Output carries a Resemble watermark.
2. **AI4Bharat IndicF5** — MIT (gated, consent clause), best-in-class Hindi
   naturalness, F5-style cloning (needs reference transcript). Rough on Macs:
   needs pinned `torch==2.2.0` + MPS fallback; RTF ~1.5–2 reported on M4 Max.
   No MLX port.
3. **Qwen3-TTS 0.6B/1.7B** — Apache-2.0, weights open since Jan 2026, MLX port
   in `mlx-audio`, 3 s cloning. Officially 10 languages **without Hindi**, but
   Hindi works in practice (above). Treat as experimental.
4. **Zonos2** (Zyphra, Jun 2026) — Apache-2.0, MoE, reference-language-agnostic
   cloning, has an mlx-audio implementation; Hindi is its lowest quality tier.
   Untested here.

Not suitable: CosyVoice 2/3 and official Qwen3-TTS language list (no Hindi),
XTTS-v2 (non-commercial CPML), IndicParler-TTS (no cloning), Voxtral TTS
(CC-BY-NC), Fish/OpenAudio (NC). Sarvam and Krutrim have released no open TTS
weights as of July 2026.

## Local translation options

1. **Gemma 4** (Apr 2026) — now **Apache-2.0**; ollama tags incl. `gemma4:12b`,
   `gemma4:26b-mlx` (MoE, ~4B active). Follows the pipeline's word-budget +
   code-mixing prompt as-is. Our default local pick.
2. **TranslateGemma 12B** — Google's dedicated MT model (Jan 2026), highest pure
   MT quality; fixed prompt template, terminology-instruction compliance
   unverified. Gemma Terms (not Apache).
3. **Qwen 3.5** (27B / 35B-A3B MoE) — Apache-2.0, 201 languages, MLX tags on ollama.
4. **Sarvam-30B** (Mar 2026) — Apache-2.0, Indic-first MoE (2.4B active), GGUF;
   no published MT benchmarks yet.
5. **IndicTrans2** (MIT) — dedicated En→Indic NMT, but no instruction channel:
   cannot follow "keep technical terms in English". IndicTrans3-beta (Gemma-3
   based) is CC-BY-4.0 and gated, still beta.

## License quick reference

Apache-2.0 / MIT (institutional use fine): Gemma 4, Qwen 3/3.5, Sarvam-30B,
Qwen3-TTS, Kokoro, IndicParler, IndicF5 (gated + consent clause), Chatterbox,
Zonos2, IndicTrans2. Restricted: XTTS-v2 (CPML non-commercial), Voxtral TTS /
Fish / Aya / NLLB (CC-BY-NC), sarvam-translate (GPL-3.0), TranslateGemma &
Gemma 3 (Gemma Terms).
