# Is translate → synthesize → time-stretch the right design?

A study of how VidVaani's segment-level alignment approach compares to the
state of the art in automatic dubbing (surveyed July 2026), and what we
changed as a result.

## The question

VidVaani splits a lecture into Whisper segments (≤ 15 s), translates each with
a duration-aware prompt, synthesizes Hindi TTS per segment, then force-fits
each clip into its original time slot with `ffmpeg atempo` speed adjustment
plus padding/trimming. Is this the best approach?

## What the literature says

The short answer: the *architecture* is right — segment-level ("utterance
isochrony") alignment with a soft length constraint at translation time and a
post-hoc fitting fallback is what production systems ship. But three findings
changed our defaults:

1. **Fix length at translation time, not audio time.** In human preference
   tests, verbosity-controlled translation beat post-hoc prosodic alignment by
   a wide margin (Tam et al., Interspeech 2022, arXiv:2112.08548; Amazon
   ICASSP 2021, arXiv:2110.03847). But *character*-count budgets are useless —
   character-isometric MT achieved the same speech overlap as unconstrained MT
   (arXiv:2302.12979). Budgets must be in spoken-duration units: words,
   syllables, or phonemes. For Hindi specifically, phoneme counts are the
   reliable unit (Sony, NAACL-Findings 2024, arXiv:2403.15469).

2. **Strict isochrony is the wrong objective.** Professional human dubs
   overlap the source speech timing only ~66% of the time and consistently
   sacrifice timing rather than naturalness or content ("Dubbing in
   Practice", TACL 2023, arXiv:2212.12137). Enforcing strict timing cost
   −10.9 MUSHRA points (arXiv:2001.06785); relaxing it for off-screen speech
   (which a slide lecture almost always is) won listener preference by 2–4×
   (arXiv:2204.02530). **Deleting content to fit a slot is the worst
   possible trade.**

3. **Uniform time-stretching is the worst way to change speech rate.**
   TTS-native, phoneme-aware duration control beat uniform stretching by
   +29% to +173% relative preference (Amazon ICASSP 2022). Perceptual limits:
   speed-ups tolerable to ~1.35–1.4×, slow-downs degrade quality much faster
   (slow variants scored ~1–1.5 points worse on a 0–10 scale in every tested
   condition), and 2.0× compression is unintelligible.

English→Hindi expands ~15–35% in word count, and stock En→Hi MT misses even a
±20% phoneme budget on ~27% of sentences — so roughly a quarter of segments
genuinely need intervention beyond gentle stretching.

## What VidVaani already got right

- Utterance-level isochrony only — no phrase/lip-sync alignment, which the
  off-screen literature says is unnecessary for lectures.
- Soft duration hint in the translation prompt (hard budgets backfire:
  −5.7 BLEU and strongly dis-preferred).
- Segments placed at absolute start times — starting on time matters more
  than ending on time.
- Padding short clips instead of slowing them further.
- Batched translation gives cross-segment context; translation and TTS caches.

## Changes implemented (July 2026)

| Change | Evidence |
|---|---|
| **Never truncate**: overlong clips spill into the trailing pause (up to the next segment's start, ≤ 2 s), truncation only at that hard cap | Humans prefer broken timing over deleted content (TACL 2023) |
| **Asymmetric speed clamp**: was 0.85×–1.25×, now 0.95×–1.35× (pad instead of slowing; more speed-up headroom before the last resort) | Slow-downs hurt more than speed-ups (ICASSP 2022); fluent envelope 0.6–1.4 (Interspeech 2020) |
| **Trim TTS trailing silence before measuring** duration | Otherwise speech gets sped up to make room for silence (open-dubbing does the same) |
| **Numeric word budget in the translation prompt** (`max_words ≈ 2.4 × duration_s`), calibrated against the measured Sarvam delivery rate (~2.8 words/s ≈ 170 wpm — much faster than human-narration figures), with an explicit "come close to the budget" instruction | Duration-unit budgets shift length; vague instructions are ignored (IWSLT 2025, arXiv:2506.04855). A first attempt at 1.8 w/s (110 wpm assumption) yielded only 61% median slot fill — audibly sparse |
| **Explicit code-mixing rule**: technical terms stay in English/transliteration | NPTEL production policy across 20,000+ hours; Sanskritized terminology rejected by students |
| Original audio muted for the clip's *actual* duration, not the nominal slot | Prevents double-audio during spill-over |
| **Final mix loudness matched to the source video** (measured via `loudnorm` print mode) instead of a fixed −16 LUFS | The fixed broadcast target made every dub ~3 LU louder than its original; A/B comparisons jumped in volume |

## Next steps (not yet implemented, ranked by value/effort)

1. **Two-pass Sarvam `pace`**: synthesize at 1.0, measure, re-synthesize at
   `pace = clamp(actual/target, 0.9, 1.35)` — native rate control beats
   waveform stretching in every published comparison. Sarvam bulbul:v2
   supports pace 0.3–3.0; cost of the second pass is trivial (₹15/10K chars).
   Keep a small atempo (0.95–1.05×) as the final trim.
2. **Re-translate outliers once**: segments still needing > 1.3× speed get one
   "shorten by ~25%, same meaning, keep English technical terms" LLM call +
   resynthesis. Bounded cost, no loop — mirrors the field's
   one-pass-plus-repair pattern. Expect ~25% of segments to qualify.
3. **Sentence-boundary segmentation**: word-level timestamps (WhisperX-style),
   group at sentence boundaries snapped to ≥ 300 ms pauses instead of blind
   ≤ 15 s packing. Sentence-like units beat VAD units by ~4.5 BLEU
   (arXiv:2202.04774).
4. **Persistent glossary across batches**: accumulate technical-term renderings
   and re-inject per batch — the largest measured document-context gain is
   terminology consistency (EMNLP 2023, arXiv:2304.02210).
5. **Instrumentation**: log per-segment speech overlap and % of segments
   saturating the speed clamp (the field's standard SO / SLC@0.2 metrics) to
   measure whether each change pays off.

Note on backends: Gemini TTS has no numeric rate or duration control
(prompt-only, officially unreliable), so the Sarvam backend is preferred for
the dubbing path. Azure's `mstts:audioduration` is the only major API with
true target-duration synthesis, if an alternative backend is ever needed.
