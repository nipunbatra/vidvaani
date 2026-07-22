---
marp: true
theme: acad
paginate: true
header: "VIDVAANI · IIT GANDHINAGAR"
title: "VidVaani — Technical lectures, heard in Hindi"
---

<!-- _class: title -->
<!-- _header: "" -->

<span class="eyebrow">OPEN-SOURCE PIPELINE · HINDI TECHNICAL LECTURES</span>

# Technical lectures<br>in Hindi.

<p class="subtitle">VidVaani converts English technical lectures into timing-aligned Hindi audio while preserving technical vocabulary and the original video.</p>

<p class="author">Nipun Batra</p>
<p class="affil">Computer Science &amp; Engineering · IIT Gandhinagar</p>
<p class="date">July 2026 · nipunbatra.github.io/vidvaani</p>

---

# Motivation

<div class="metrics">
<div class="metric"><span class="value">50,000+</span><span class="label">hours of NPTEL engineering lectures</span></div>
<div class="metric"><span class="value">1 command</span><span class="label">from a YouTube link or local video to a Hindi dub</span></div>
<div class="metric"><span class="value">₹7–11</span><span class="label">measured cost for a seven-minute lecture</span></div>
</div>

<p class="big-line">For many students, spoken English limits how effectively a technical lecture can be reused. The aim is a <span class="punch">low-cost Hindi version that remains faithful to the teacher’s terminology and timing.</span></p>

<div class="foot">NPTEL catalogue and translation initiative · Measured VidVaani runs, July 2026.</div>

---

# Limitations of existing options

<div class="cards">
<div class="card"><span class="tag">PLATFORM</span><h2>YouTube</h2><p>YouTube determines channel eligibility. Nipun’s lecture channel is not enabled. Dubs cannot be edited, and instructors cannot choose terminology or voice.</p></div>
<div class="card"><span class="tag">SAAS</span><h2>Commercial services</h2><p>Typical prices are ₹3,000–11,000 per lecture, often with monthly limits and additional charges for each language.</p></div>
<div class="card"><span class="tag">HUMAN WORKFLOW</span><h2>NPTEL / Bhashini</h2><p>These programmes provide careful review, but operate course by course and are not self-service tools for instructors.</p></div>
</div>

> Need: a self-service system with explicit control over voice, vocabulary, timing, and cost.

<div class="foot">YouTube eligibility and limitations: support.google.com/youtube/answer/15569972 · Vendor prices surveyed July 2026.</div>

---

# System overview

<div class="figure-surface">

![w:1080](figures/pipeline.svg)

</div>

<div class="metrics">
<div class="metric"><span class="value">Local STT</span><span class="label">MLX Whisper performs transcription on-device</span></div>
<div class="metric"><span class="value">Cached</span><span class="label">a new voice can reuse earlier results</span></div>
<div class="metric"><span class="value">Measured</span><span class="label">time, tokens, cost and output metadata are recorded</span></div>
</div>

---

# Inspecting a completed run

![class:shot](figures/control_room.png)

<p class="cap">Each phase reports its status, elapsed time, model, cost and intermediate outputs.</p>

---

# Translation for a technical classroom

<div class="pair">
<time>00:00–00:15</time>
<div><span class="lang">ENGLISH</span><p>Today I want to show the usage of a tool called GeoGebra.</p></div>
<div class="hindi"><span class="lang">HINDI</span><p>आज मैं जियोजेब्रा नामक एक टूल का उपयोग दिखाना चाहता हूँ।</p></div>
</div>

<div class="pair">
<time>00:28–00:40</time>
<div><span class="lang">ENGLISH</span><p>The x range is between zero to two and the y range is between zero to two.</p></div>
<div class="hindi"><span class="lang">HINDI</span><p>x और y की रेंज शून्य से दो तक है।</p></div>
</div>

<p class="big-line">Terms such as <strong>GeoGebra, x, y and PDF</strong> are retained. The explanation is translated into natural spoken Hindi.</p>

---

# Timing and assembly

<div class="figure-surface">

![w:860](figures/waveform_compare.png)

</div>

<p class="big-line">Each translation receives a word budget. After synthesis, the clip is trimmed and its pace is adjusted; original audio remains in the gaps.</p>

---

# Demo runs stop at the requested segment

<div class="metrics">
<div class="metric"><span class="value">5</span><span class="label">requested segments</span></div>
<div class="metric"><span class="value">58.5 s</span><span class="label">final duration, with no English remainder</span></div>
<div class="metric"><span class="value">1864×1080</span><span class="label">source resolution preserved</span></div>
</div>

<p class="big-line">Demo mode ends after the last generated Hindi clip. Full mode processes the complete source. The original video stream is copied without reducing its resolution.</p>

<pre><code>uv run --frozen vidvaani-web
# “Explore completed sample” makes no API calls</code></pre>

---

# Measured time and cost

<div class="cols">
<div class="figure-surface">

![w:510](figures/timing_breakdown.png)

</div>
<div class="figure-surface">

![w:510](figures/cost_comparison.png)

</div>
</div>

<div class="metrics">
<div class="metric"><span class="value">2½–4 min</span><span class="label">end-to-end for a seven-minute lecture</span></div>
<div class="metric"><span class="value">₹80–110</span><span class="label">extrapolated per lecture-hour</span></div>
<div class="metric"><span class="value">₹0</span><span class="label">for a cached no-API demo run</span></div>
</div>

---

# Cloud and local speech options

<div class="cards">
<div class="card"><span class="tag">CLOUD</span><h2>Sarvam</h2><p>Clear Hindi with Indian prosody and several voice choices. This is the default demonstration backend.</p></div>
<div class="card"><span class="tag">CLOUD</span><h2>Gemini TTS</h2><p>Natural delivery and style control, although its Hindi accent may sound less local.</p></div>
<div class="card"><span class="tag">LOCAL · ₹0</span><h2>Gemma 4 + Qwen3-TTS</h2><p>Translation and speech run on-device. Lecturer voice cloning is used only with consent.</p></div>
</div>

<div class="metrics">
<div class="metric"><span class="value">59 s</span><span class="label">Gemma 4 31B translation on the 58 s clip</span></div>
<div class="metric"><span class="value">1.6× RTF</span><span class="label">Qwen3-TTS cloned speech generation</span></div>
<div class="metric"><span class="value">0.89</span><span class="label">best consented speaker similarity; STT-checked</span></div>
</div>

<div class="foot">Experimental local chain: experiments/local_prob/step1_transcribe.py → step4_assemble.py · Method and results: docs/local-models.md.</div>

---

# Demonstrations

<div class="cols">
<div>

<span class="eyebrow">NIPUNBATRA.GITHUB.IO/VIDVAANI</span>

## Available comparisons

- Original English beside the Hindi result
- Sarvam, Gemini, fully local, and cloned-voice paths
- Short probability demo plus a full NPTEL lecture
- Methods, costs, privacy and limitations documented alongside the examples

<p class="big-line">The website supports audio comparison; the control room shows how each output was produced.</p>

</div>
<div>

![w:450](figures/demo_frame.jpg)

![w:145](figures/qr_demo.png)

</div>
</div>

---

# Limitations and next steps

<div class="cards">
<div class="card"><span class="tag">QUALITY</span><h2>Faculty review</h2><p>The system produces a draft; subject experts must still review its pedagogical accuracy.</p></div>
<div class="card"><span class="tag">ALIGNMENT</span><h2>Dense passages</h2><p>Some segments require a shorter translation rather than faster synthesized speech.</p></div>
<div class="card"><span class="tag">TRUST</span><h2>Voice consent</h2><p>Voice cloning is opt-in, and speaker similarity alone is not sufficient for release.</p></div>
</div>

<p class="big-line">Next steps are faculty-in-the-loop editing, a terminology glossary, native TTS pace control, support for more Indian languages, and evaluation of student comprehension.</p>

---

<!-- _class: close -->
<!-- _header: "" -->

<span class="eyebrow">VIDVAANI</span>

# From an English lecture<br>to a Hindi version.

<p class="big-line">The pipeline is open, its intermediate results can be inspected, and the translation and speech stages can run locally when required.</p>

<div class="metrics">
<div class="metric"><span class="value">Demo</span><span class="label">nipunbatra.github.io/vidvaani</span></div>
<div class="metric"><span class="value">Code</span><span class="label">github.com/nipunbatra/vidvaani</span></div>
<div class="metric"><span class="value">Question</span><span class="label">Which lecture should be evaluated next?</span></div>
</div>
