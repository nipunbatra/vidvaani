# VidVaani Browser Lab

An experimental, static, local-first Hindi lecture-dubbing pipeline hosted entirely on GitHub Pages. No VidVaani server receives the user's video, audio, transcript, translation, API key, or output.

This is deliberately a short-clip research build. It accepts clips up to 120 seconds so that model downloads, browser memory use, and assembly remain visible and recoverable during a live demonstration.

## Pipeline

1. Inspect the selected media in the browser.
2. Extract 16 kHz mono audio with FFmpeg.wasm.
3. Transcribe English with Whisper through Transformers.js.
4. Translate timed segments with local Qwen3 or optional Gemini.
5. Synthesize Hindi with local MMS-TTS or optional Gemini TTS.
6. Align the speech clips to their original segment start times and assemble the output locally.
7. Export the dubbed media, Hindi subtitles, and an inspectable JSON run report.

Every phase has a stable status row, elapsed time, relevant statistics, and phase-specific evidence. Model downloads are reported separately from inference.

## Model selection

| Task | Default | Larger / cloud option | Why |
|---|---|---|---|
| English ASR | `onnx-community/whisper-base` | `whisper-small.en` | Base is the practical accuracy/download compromise; Tiny is available for weaker devices. WebGPU is preferred, with WASM fallback. |
| EN → HI translation | `onnx-community/Qwen3-0.6B-ONNX`, q4f16 | Qwen3 1.7B q4f16 or Gemini 2.5 Flash | 0.6B is approximately 570 MB and is the smallest credible instruction-following option tested for the timed JSON task. Local translation requires WebGPU. |
| Hindi speech | `Xenova/mms-tts-hin`, q8 | fp32 or Gemini 2.5 Flash preview TTS | MMS is small enough for a WASM browser path. The hosted model is research/non-commercial; check its licence before deployment. |
| Media | FFmpeg.wasm | — | Extraction, mixing, timing, and muxing stay on the device. |

Gemma 4 31B and Qwen3-TTS 1.7B remain part of VidVaani's native MLX pipeline, not this browser build. Their size and current browser runtime support make them inappropriate defaults for a dependable public WebGPU demo.

## Gemini expert mode

The optional Gemini path is direct browser-to-Google BYOK. It exists because this is an inspectable static research prototype, not because browser-side production API keys are generally safe.

- The key is kept in JavaScript memory only.
- It is never written to local storage, IndexedDB, URLs, logs, artifacts, or analytics.
- Requests use the `x-goog-api-key` header, `credentials: omit`, `cache: no-store`, and `referrerPolicy: no-referrer`.
- Only timed text is sent for translation; only Hindi text is sent for speech. The source media is never sent to Gemini.
- The key is erased from page state and the password input after every run, including failed and cancelled runs.
- Users should supply a disposable, API-restricted key and revoke it after the demonstration. A durable public product should use a controlled backend or short-lived brokered credentials.

See [SECURITY.md](SECURITY.md) for the threat model and GitHub Pages limitations.

## Build and test

```bash
npm install
npm test
npm run build
```

Vite writes the deployable static site to the repository's `lab/` directory using relative asset paths, so it works at `/vidvaani/lab/` on GitHub Pages and from a local static server.

For development:

```bash
npm run dev
```

Use an HTTPS origin or `localhost`. WebGPU support and practical model capacity vary by browser, operating system, driver, and available memory.

## Privacy boundary

Project media and generated artifacts last only for the page session. Browser object URLs are revoked on replacement or unload. Model weights can remain in the browser's ordinary HTTP/cache storage so later runs do not necessarily redownload them; users can clear site data to remove them.

The page has no analytics, service worker, cookies, form submission, remote fonts, or third-party executable scripts. Model weights are fetched from explicitly allow-listed Hugging Face delivery origins. Executable JavaScript, WebAssembly runtimes, and FFmpeg core files are part of the same-origin build.

## Primary implementation references

- [Transformers.js WebGPU guide](https://huggingface.co/docs/transformers.js/en/guides/webgpu)
- [Transformers.js automatic speech recognition task](https://huggingface.co/docs/transformers.js/en/api/pipelines#module_pipelines.AutomaticSpeechRecognitionPipeline)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [MMS Hindi Transformers.js model card](https://huggingface.co/Xenova/mms-tts-hin)
- [FFmpeg.wasm documentation](https://ffmpegwasm.netlify.app/docs/overview/)
- [Google API key security guidance](https://support.google.com/googleapi/answer/6310037)
