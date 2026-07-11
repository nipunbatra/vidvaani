# Going fully local: open-weights models for the pipeline

Survey (July 2026) plus measured experiments on Apple Silicon. Goal: replace
the two cloud stages (translation, TTS) with local open-weights models.

## Measured results (58 s demo clip, M-series Mac)

| Stage | Model | Time | Notes |
|---|---|---|---|
| Translate | Gemma 4 12B via ollama | 24.5 s (5 segments) | Quality close to Gemini 2.5 Flash; needed one extra prompt rule ("spoken words only — no LaTeX/parentheses"); even corrected a Whisper mishearing |
| Translate | Gemma 3 12B via ollama | 26.4 s | Slightly weaker phrasing, still usable |
| TTS (default voice) | Qwen3-TTS 1.7B (MLX, `mlx-audio`) | RTF ≈ 4.4 | Hindi is NOT in its official language list but works empirically (verified by Sarvam STT round-trip) |
| TTS (cloned voice) | Qwen3-TTS + 30 s English reference | RTF ≈ 4.0 | Cross-lingual cloning: English reference → same voice speaking Hindi, word-perfect STT round-trip |
| Model load | Qwen3-TTS | 21.8 s once | |

The "Fully local" card on the demo page = local Whisper + Gemma 4 12B +
Qwen3-TTS (default voice), assembled by the normal pipeline. ₹0 per lecture;
a 1-hour lecture is roughly an overnight batch at RTF ~4–5.

**Voice cloning & consent:** the cloned-voice output is generated from a
lecture by a third-party professor and is therefore NOT published; it lives in
`experiments/voice_cloning/output/local_cloned_demo.mp4` for internal
evaluation only. Publish cloned voices only with the speaker's consent — the
compelling legitimate use is a lecturer dubbing their own course in their own
voice.

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
