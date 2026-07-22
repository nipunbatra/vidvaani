import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
  pipeline,
} from "@huggingface/transformers";
import ortAsyncFactoryURL from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortAsyncWasmURL from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import ortFactoryURL from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasmURL from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

// Keep executable runtime code on the same GitHub Pages origin. Model weights
// are the only remote assets and are fetched from the allow-listed HF origin.
const safari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
env.backends.onnx.wasm.wasmPaths = safari
  ? { mjs: ortFactoryURL, wasm: ortWasmURL }
  : { mjs: ortAsyncFactoryURL, wasm: ortAsyncWasmURL };
env.backends.onnx.wasm.proxy = false;

let transcriber = null;
let transcriberKey = "";
let translator = null;
let translatorTokenizer = null;
let translatorKey = "";
let synthesizer = null;
let synthesizerKey = "";

function report(requestId, status, data = {}) {
  self.postMessage({ requestId, status, ...data });
}

function modelProgress(requestId, task) {
  return (event) => {
    report(requestId, "model-progress", {
      task,
      file: event.file ?? "model data",
      progress: Number.isFinite(event.progress) ? event.progress : null,
      loaded: event.loaded ?? null,
      total: event.total ?? null,
    });
  };
}

async function getTranscriber(requestId, modelId, useWebGPU) {
  const device = useWebGPU ? "webgpu" : "wasm";
  const key = `${modelId}:${device}`;
  if (transcriber && transcriberKey !== key) {
    await transcriber.dispose?.();
    transcriber = null;
  }
  if (!transcriber) {
    transcriberKey = key;
    report(requestId, "loading", { task: "transcription", message: `Loading ${modelId}` });
    transcriber = await pipeline("automatic-speech-recognition", modelId, {
      device,
      dtype: useWebGPU
        ? { encoder_model: "fp32", decoder_model_merged: "q4" }
        : "q8",
      progress_callback: modelProgress(requestId, "transcription"),
    });
  }
  return transcriber;
}

async function transcribe(requestId, payload) {
  const pipe = await getTranscriber(requestId, payload.modelId, payload.useWebGPU);
  report(requestId, "running", { task: "transcription", message: "Decoding English speech" });
  const options = {
    chunk_length_s: 30,
    stride_length_s: 5,
    return_timestamps: true,
  };
  if (!payload.modelId.endsWith(".en")) {
    options.language = "en";
    options.task = "transcribe";
  }
  const result = await pipe(payload.audio, options);
  report(requestId, "complete", { task: "transcription", result });
}

async function getTranslator(requestId, modelId) {
  if ((translator || translatorTokenizer) && translatorKey !== modelId) {
    await translator?.dispose?.();
    translator = null;
    translatorTokenizer = null;
  }
  if (!translator || !translatorTokenizer) {
    translatorKey = modelId;
    report(requestId, "loading", { task: "translation", message: `Loading ${modelId}` });
    const progress = modelProgress(requestId, "translation");
    [translatorTokenizer, translator] = await Promise.all([
      AutoTokenizer.from_pretrained(modelId, { progress_callback: progress }),
      AutoModelForCausalLM.from_pretrained(modelId, {
        device: "webgpu",
        dtype: "q4f16",
        progress_callback: progress,
      }),
    ]);
  }
  return [translatorTokenizer, translator];
}

async function translate(requestId, payload) {
  const [tokenizer, model] = await getTranslator(requestId, payload.modelId);
  const results = [];
  for (let index = 0; index < payload.segments.length; index += 1) {
    const segment = payload.segments[index];
    const maxWords = Math.max(4, Math.floor((segment.end - segment.start) * 2.4));
    report(requestId, "segment", {
      task: "translation",
      index,
      total: payload.segments.length,
      message: `Translating segment ${index + 1} of ${payload.segments.length}`,
      text: segment.original,
    });
    let translated = await generateTranslation(tokenizer, model, segment.original, maxWords, false);
    if (devanagariRatio(translated) < 0.45) {
      report(requestId, "running", {
        task: "translation",
        message: `Correcting script for segment ${index + 1}`,
      });
      translated = await generateTranslation(tokenizer, model, segment.original, maxWords, true);
    }
    if (!translated) throw new Error(`Local translation was empty for segment ${index + 1}.`);
    results.push({ ...segment, translated });
    report(requestId, "stream", {
      task: "translation",
      text: translated,
      index,
      total: payload.segments.length,
    });
  }
  report(requestId, "complete", { task: "translation", result: results });
}

async function generateTranslation(tokenizer, model, source, maxWords, strictScript) {
  const messages = [
    {
      role: "system",
      content: strictScript
        ? "Translate into natural spoken Hindi. Write only Devanagari. Transliterate every English technical term into Devanagari. Return only the translation."
        : "Translate English technical lectures into natural spoken Hindi. Write Hindi in Devanagari and transliterate technical terms into Devanagari so a Hindi speech model can read them. Return the translation only, with no note or label.",
    },
    { role: "user", content: `Translate in at most ${maxWords} words: ${source}` },
  ];
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  });
  let outputText = "";
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => { outputText += text; },
  });
  await model.generate({
    ...inputs,
    do_sample: false,
    max_new_tokens: Math.min(160, Math.max(48, maxWords * 4)),
    repetition_penalty: 1.05,
    streamer,
  });
  return cleanTranslation(outputText);
}

function cleanTranslation(value) {
  return String(value)
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/^```(?:\w+)?|```$/g, "")
    .replace(/^(?:Hindi|Translation|अनुवाद)\s*:\s*/i, "")
    .trim()
    .replace(/^['\"]|['\"]$/g, "");
}

function devanagariRatio(value) {
  const letters = String(value).match(/\p{L}/gu) ?? [];
  if (!letters.length) return 0;
  return letters.filter((letter) => /[\u0900-\u097F]/.test(letter)).length / letters.length;
}

async function getSynthesizer(requestId, dtype) {
  const key = `Xenova/mms-tts-hin:${dtype}`;
  if (synthesizer && synthesizerKey !== key) {
    await synthesizer.dispose?.();
    synthesizer = null;
  }
  if (!synthesizer) {
    synthesizerKey = key;
    report(requestId, "loading", { task: "speech", message: "Loading MMS Hindi" });
    synthesizer = await pipeline("text-to-speech", "Xenova/mms-tts-hin", {
      device: "wasm",
      dtype,
      progress_callback: modelProgress(requestId, "speech"),
    });
  }
  return synthesizer;
}

async function synthesize(requestId, payload) {
  const pipe = await getSynthesizer(requestId, payload.dtype);
  const outputs = [];
  for (let index = 0; index < payload.segments.length; index += 1) {
    const segment = payload.segments[index];
    report(requestId, "segment", {
      task: "speech",
      index,
      total: payload.segments.length,
      message: `Synthesizing segment ${index + 1} of ${payload.segments.length}`,
    });
    const speakable = prepareHindiSpeech(segment.translated);
    if (!/[\u0900-\u097F]/.test(speakable)) {
      throw new Error(`Segment ${index + 1} contains no speakable Devanagari text.`);
    }
    const result = await pipe(speakable);
    const raw = result.audio?.data ?? result.audio;
    const audio = Float32Array.from(raw);
    outputs.push({
      start: segment.start,
      end: segment.end,
      audio,
      samplingRate: result.sampling_rate,
    });
  }
  report(requestId, "complete", { task: "speech", result: outputs });
}

function prepareHindiSpeech(value) {
  const replacements = new Map([
    ["pdf", "पीडीएफ"], ["psdv", "पीएसडीवी"], ["geogebra", "जिओजेब्रा"],
    ["x", "एक्स"], ["y", "वाई"], ["z", "ज़ेड"], ["function", "फंक्शन"],
    ["integral", "इंटीग्रल"], ["integrals", "इंटीग्रल्स"], ["dimension", "डाइमेंशन"],
    ["dimensions", "डाइमेंशन्स"], ["zero", "ज़ीरो"], ["hello", "नमस्ते"],
  ]);
  return String(value)
    .normalize("NFC")
    .replace(/\uFFFD/g, "")
    .replace(/[A-Za-z][A-Za-z0-9._+-]*/g, (word) => replacements.get(word.toLowerCase()) ?? " ")
    .replace(/[^\u0900-\u097F0-9\s.,?!:;()\-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

self.addEventListener("message", async (event) => {
  const { requestId, type, payload } = event.data;
  try {
    if (type === "transcribe") await transcribe(requestId, payload);
    else if (type === "translate") await translate(requestId, payload);
    else if (type === "synthesize") await synthesize(requestId, payload);
    else throw new Error("Unknown worker operation");
  } catch (error) {
    report(requestId, "error", {
      message: error instanceof Error ? error.message : "Local model operation failed",
    });
  }
});
