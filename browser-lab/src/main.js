import "./styles.css";
import { ModelClient } from "./model-client.js";
import {
  MediaEngine,
  decodeWav,
  floatAudioToWav,
  pcm16ToWav,
  probeMedia,
} from "./media.js";
import { synthesizeWithGemini, translateWithGemini } from "./gemini.js";

const MAX_DURATION_SECONDS = 120;
const MAX_FILE_BYTES = 400 * 1024 * 1024;
const USD_TO_INR = 95.4;

const PHASE_DEFINITIONS = [
  ["inspect", "Media inspection", "Validate the file, duration, format and local privacy boundary."],
  ["extract", "Audio extraction", "Decode a 16 kHz mono track locally with FFmpeg.wasm."],
  ["transcribe", "English transcript", "Run Whisper locally and retain timed English segments."],
  ["translate", "Hindi translation", "Translate to spoken Hindi while preserving technical vocabulary."],
  ["speech", "Hindi speech", "Generate one Hindi audio clip for each translated segment."],
  ["assemble", "Timing and assembly", "Place speech at source timestamps and retain the original video stream."],
  ["deliver", "Local artifacts", "Expose MP4 or WAV, SRT and JSON as temporary browser URLs."],
];

const elements = Object.fromEntries(
  [
    "runtime-pill", "runtime-label", "media-file", "dropzone", "file-summary", "load-demo",
    "cap-secure", "cap-webgpu", "cap-wasm", "cap-memory", "stt-model", "stt-help",
    "translation-model", "translation-help", "speech-model", "speech-help",
    "key-panel", "gemini-key", "forget-key", "run-button", "cancel-button",
    "form-error", "phase-list", "phase-detail", "detail-eyebrow", "detail-title",
    "detail-description", "detail-time", "detail-progress", "detail-metrics",
    "detail-evidence", "run-time", "activity-list", "clear-activity",
    "outputs-section", "output-video", "output-actions",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  file: null,
  metadata: null,
  capabilities: { secure: false, webgpu: false, wasm: false, memory: null },
  phases: Object.fromEntries(PHASE_DEFINITIONS.map(([id]) => [id, freshPhase()])),
  selectedPhase: "inspect",
  currentPhase: null,
  running: false,
  cancelled: false,
  runStartedAt: null,
  timer: null,
  geminiKey: "",
  abortController: null,
  activity: [],
  artifacts: [],
  segments: [],
  translatedSegments: [],
  usage: freshUsage(),
};

const mediaEngine = new MediaEngine();
let modelClient = new ModelClient(handleModelEvent);

function freshPhase() {
  return {
    status: "waiting",
    progress: 0,
    seconds: null,
    metrics: [],
    evidence: { type: "empty" },
  };
}

function freshUsage() {
  return { translationInput: 0, translationOutput: 0, ttsInput: 0, ttsOutput: 0 };
}

function initialize() {
  renderPhaseButtons();
  renderDetail();
  bindEvents();
  updateProviderUI();
  detectCapabilities();
}

function bindEvents() {
  elements["media-file"].addEventListener("change", (event) => selectFile(event.target.files?.[0]));
  elements["load-demo"].addEventListener("click", loadDemoClip);
  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragover");
  });
  elements.dropzone.addEventListener("dragleave", () => elements.dropzone.classList.remove("dragover"));
  elements.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragover");
    selectFile(event.dataTransfer?.files?.[0]);
  });
  document.querySelectorAll('input[name="translation-provider"], input[name="speech-provider"]')
    .forEach((input) => input.addEventListener("change", updateProviderUI));
  elements["stt-model"].addEventListener("change", updateModelHelp);
  elements["translation-model"].addEventListener("change", updateModelHelp);
  elements["speech-model"].addEventListener("change", updateModelHelp);
  elements["gemini-key"].addEventListener("input", (event) => {
    state.geminiKey = event.target.value.trim();
    validateReadyState();
  });
  elements["forget-key"].addEventListener("click", forgetKey);
  elements["run-button"].addEventListener("click", runPipeline);
  elements["cancel-button"].addEventListener("click", cancelRun);
  elements["clear-activity"].addEventListener("click", () => {
    state.activity = [];
    renderActivity();
  });
  window.addEventListener("beforeunload", cleanupSensitiveState);
}

async function loadDemoClip() {
  const button = elements["load-demo"];
  const original = button.firstChild.textContent;
  button.disabled = true;
  button.firstChild.textContent = "Loading lecture sample…";
  clearError();
  try {
    const response = await fetch("../demo_videos/mini_demo/english_original.mp4", {
      cache: "force-cache",
      credentials: "omit",
    });
    if (!response.ok) throw new Error(`Sample download failed with status ${response.status}.`);
    const file = new File([await response.blob()], "IITGN_probability_lecture_sample.mp4", {
      type: "video/mp4",
    });
    await selectFile(file);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The sample clip could not be loaded.");
  } finally {
    button.disabled = false;
    button.firstChild.textContent = original;
  }
}

async function detectCapabilities() {
  state.capabilities.secure = window.isSecureContext;
  state.capabilities.wasm = typeof WebAssembly === "object";
  state.capabilities.memory = navigator.deviceMemory ?? null;
  try {
    state.capabilities.webgpu = Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    state.capabilities.webgpu = false;
  }
  setCapability("cap-secure", state.capabilities.secure, state.capabilities.secure ? "Available" : "Required");
  setCapability("cap-webgpu", state.capabilities.webgpu, state.capabilities.webgpu ? "Available" : "Unavailable");
  setCapability("cap-wasm", state.capabilities.wasm, state.capabilities.wasm ? "Available" : "Unavailable");
  elements["cap-memory"].textContent = state.capabilities.memory ? `${state.capabilities.memory} GB reported` : "Not reported";
  const ready = state.capabilities.secure && state.capabilities.wasm;
  elements["runtime-pill"].classList.toggle("ready", ready);
  elements["runtime-pill"].classList.toggle("error", !ready);
  elements["runtime-label"].textContent = ready
    ? (state.capabilities.webgpu ? "WebGPU and WASM ready" : "WASM ready · no WebGPU")
    : "Secure browser context required";
  validateReadyState();
}

function setCapability(id, good, text) {
  elements[id].textContent = text;
  elements[id].className = good ? "good" : "bad";
}

async function selectFile(file) {
  clearError();
  clearArtifacts();
  if (!file) return;
  if (file.size > MAX_FILE_BYTES) return showError("This research build limits files to 400 MB.");
  if (!file.type.startsWith("video/") && !file.type.startsWith("audio/")) {
    return showError("Choose a supported video or audio file.");
  }
  try {
    const metadata = await probeMedia(file);
    if (!Number.isFinite(metadata.duration) || metadata.duration <= 0) throw new Error("Media duration is unavailable.");
    if (metadata.duration > MAX_DURATION_SECONDS) {
      throw new Error(`Choose a clip of ${MAX_DURATION_SECONDS} seconds or less for this browser prototype.`);
    }
    state.file = file;
    state.metadata = metadata;
    resetPhases(false);
    state.phases.inspect = {
      status: "complete",
      progress: 100,
      seconds: 0,
      metrics: [
        ["Duration", formatDuration(metadata.duration)],
        ["File size", formatBytes(file.size)],
        ["Resolution", metadata.width ? `${metadata.width}×${metadata.height}` : "Audio only"],
        ["MIME type", file.type || "Unknown"],
      ],
      evidence: { type: "file", file, metadata },
    };
    elements["file-summary"].textContent = `${file.name} · ${formatDuration(metadata.duration)} · ${formatBytes(file.size)}`;
    addActivity("inspect", "File validated locally");
    selectPhase("inspect");
  } catch (error) {
    state.file = null;
    state.metadata = null;
    elements["media-file"].value = "";
    elements["file-summary"].textContent = "No valid file selected";
    showError(error.message);
  }
  renderPhaseButtons();
  renderDetail();
  validateReadyState();
}

function provider(name) {
  return document.querySelector(`input[name="${name}-provider"]:checked`)?.value ?? "local";
}

function updateProviderUI() {
  const translationProvider = provider("translation");
  const speechProvider = provider("speech");
  if (translationProvider === "local") {
    setOptions(elements["translation-model"], [
      ["onnx-community/Qwen3-0.6B-ONNX", "Qwen3 0.6B q4 · fast demo"],
      ["onnx-community/Qwen3-1.7B-ONNX", "Qwen3 1.7B q4 · better Hindi"],
    ]);
    elements["translation-model"].disabled = false;
    elements["translation-help"].textContent = "Runs through WebGPU; approximately 570 MB for the recommended model.";
  } else {
    setOptions(elements["translation-model"], [["gemini-2.5-flash", "Gemini 2.5 Flash"]]);
    elements["translation-model"].disabled = true;
    elements["translation-help"].textContent = "Only timed English text is sent to Google; the media file stays local.";
  }
  if (speechProvider === "local") {
    setOptions(elements["speech-model"], [
      ["q8", "MMS Hindi q8 · recommended"],
      ["fp32", "MMS Hindi fp32 · larger model"],
    ]);
    elements["speech-help"].textContent = "Downloadable Hindi speech via Transformers.js; research/non-commercial model licence.";
  } else {
    setOptions(elements["speech-model"], [
      ["Charon", "Charon · informative"],
      ["Kore", "Kore · firm"],
      ["Aoede", "Aoede · breezy"],
      ["Iapetus", "Iapetus · clear"],
    ]);
    elements["speech-help"].textContent = "Hindi text is sent directly to Gemini TTS; raw PCM audio returns to this tab.";
  }
  elements["key-panel"].hidden = translationProvider !== "gemini" && speechProvider !== "gemini";
  updateModelHelp();
  validateReadyState();
}

function updateModelHelp() {
  const stt = {
    "onnx-community/whisper-tiny.en": "Fastest cold start for a live demo; use for clear English and weaker devices.",
    "onnx-community/whisper-base": "Balanced accuracy for clear lecture speech; WebGPU preferred, WASM fallback.",
    "onnx-community/whisper-small.en": "Best local ASR option here, with a much larger download and memory footprint.",
  };
  elements["stt-help"].textContent = stt[elements["stt-model"].value] ?? "Local Whisper transcription.";

  if (provider("translation") === "local") {
    elements["translation-help"].textContent = elements["translation-model"].value.includes("1.7B")
      ? "Higher-quality local Hindi for capable GPUs; expect a large cold download and more memory use."
      : "Fast experimental preview; approximately 570 MB. Use Qwen3 1.7B when translation quality matters.";
  }
  if (provider("speech") === "local") {
    elements["speech-help"].textContent = elements["speech-model"].value === "fp32"
      ? "Full-precision stock Hindi voice; larger and slower. Research/non-commercial model licence."
      : "Compact stock Hindi voice through WASM; fastest local option. Research/non-commercial licence.";
  } else {
    const voices = {
      Charon: "Informative, measured delivery for explanatory material.",
      Kore: "Firm, direct delivery with more authority.",
      Aoede: "Lighter, breezier delivery for accessible narration.",
      Iapetus: "Clear, even delivery suited to technical walkthroughs.",
    };
    elements["speech-help"].textContent = `${voices[elements["speech-model"].value]} Hindi text is sent directly to Gemini TTS.`;
  }
}

function setOptions(select, options) {
  select.replaceChildren(...options.map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
}

function validateReadyState() {
  let error = "";
  if (!state.capabilities.secure) error = "Open this app over HTTPS or localhost.";
  else if (!state.capabilities.wasm) error = "This browser does not support WebAssembly.";
  else if (!state.file || !state.metadata) error = "Choose a short lecture clip.";
  else if (provider("translation") === "local" && !state.capabilities.webgpu) {
    error = "Local Qwen translation requires WebGPU; select Gemini on this device.";
  } else if ((provider("translation") === "gemini" || provider("speech") === "gemini") && !state.geminiKey) {
    error = "Enter a Gemini API key for the selected cloud phase.";
  }
  elements["run-button"].disabled = Boolean(error) || state.running;
  if (!state.running) elements["form-error"].textContent = error;
  return !error;
}

async function runPipeline() {
  if (!validateReadyState()) return;
  const usedGemini = provider("translation") === "gemini" || provider("speech") === "gemini";
  clearError();
  clearArtifacts();
  resetPhases(true);
  state.running = true;
  state.cancelled = false;
  state.usage = freshUsage();
  state.abortController = new AbortController();
  state.runStartedAt = performance.now();
  elements["cancel-button"].hidden = false;
  elements["run-button"].disabled = true;
  startTimer();

  let inputName;
  try {
    state.phases.inspect.status = "complete";
    state.phases.inspect.progress = 100;
    renderPhaseButtons();

    const extraction = await runPhase("extract", async () => {
      const result = await mediaEngine.extractAudio(state.file, (progress) => {
        updatePhase("extract", {
          progress: clampProgress(progress),
          evidence: { type: "progress", title: "Extracting 16 kHz mono audio", text: "FFmpeg.wasm is operating in this tab." },
        });
      });
      inputName = result.inputName;
      const audio = await decodeWav(result.audioBytes);
      updatePhase("extract", {
        metrics: [
          ["Sample rate", "16,000 Hz"],
          ["Channels", "Mono"],
          ["Samples", audio.length.toLocaleString()],
          ["Network", "None"],
        ],
        evidence: { type: "progress", title: "Audio ready", text: `${formatDuration(audio.length / 16000)} decoded for local transcription.` },
      });
      return audio;
    });

    const transcriptResult = await runPhase("transcribe", async () => {
      const result = await modelClient.transcribe(
        extraction,
        elements["stt-model"].value,
        state.capabilities.webgpu,
      );
      const segments = normalizeTranscript(result, state.metadata.duration);
      if (!segments.length) throw new Error("Whisper returned no speech segments.");
      state.segments = segments;
      updatePhase("transcribe", {
        metrics: [
          ["Segments", String(segments.length)],
          ["Model", shortModel(elements["stt-model"].value)],
          ["Device", state.capabilities.webgpu ? "WebGPU" : "WASM"],
          ["Uploaded", "0 bytes"],
        ],
        evidence: { type: "transcript", segments },
      });
      return segments;
    });

    const translated = await runPhase("translate", async () => {
      let items;
      if (provider("translation") === "gemini") {
        const prompt = translationPrompt(transcriptResult);
        const response = await translateWithGemini(prompt, state.geminiKey, state.abortController.signal);
        items = parseTranslation(response.text, transcriptResult);
        addGeminiUsage("translation", response.usage);
      } else {
        items = await modelClient.translate(transcriptResult, elements["translation-model"].value);
      }
      state.translatedSegments = items;
      const cost = estimateCost();
      updatePhase("translate", {
        metrics: [
          ["Segments", String(items.length)],
          ["Provider", provider("translation") === "local" ? shortModel(elements["translation-model"].value) : "Gemini 2.5 Flash"],
          ["Cost", provider("translation") === "local" ? "$0 · ₹0" : `${usd(cost.translation)} · ${inr(cost.translation)}`],
          ["Video sent", "No"],
        ],
        evidence: { type: "translation", segments: items },
      });
      return items;
    });

    const speechSegments = await runPhase("speech", async () => {
      let outputs;
      if (provider("speech") === "local") {
        const local = await modelClient.synthesize(translated, elements["speech-model"].value);
        outputs = local.map((item) => ({
          start: item.start,
          end: item.end,
          blob: floatAudioToWav(item.audio, item.samplingRate),
        }));
      } else {
        outputs = [];
        for (let index = 0; index < translated.length; index += 1) {
          ensureNotCancelled();
          updatePhase("speech", {
            progress: Math.round((index / translated.length) * 90) + 5,
            evidence: { type: "progress", title: `Gemini speech ${index + 1} of ${translated.length}`, text: translated[index].translated },
          });
          const result = await synthesizeWithGemini(
            translated[index].translated,
            state.geminiKey,
            elements["speech-model"].value,
            state.abortController.signal,
          );
          addGeminiUsage("tts", result.usage);
          const rate = Number(result.mimeType.match(/rate=(\d+)/i)?.[1] ?? 24000);
          outputs.push({
            start: translated[index].start,
            end: translated[index].end,
            blob: pcm16ToWav(result.pcm, rate),
          });
        }
      }
      const cost = estimateCost();
      updatePhase("speech", {
        metrics: [
          ["Clips", String(outputs.length)],
          ["Provider", provider("speech") === "local" ? "MMS Hindi" : `Gemini ${elements["speech-model"].value}`],
          ["Cost", provider("speech") === "local" ? "$0 · ₹0" : `${usd(cost.tts)} · ${inr(cost.tts)}`],
          ["Media sent", "No"],
        ],
        evidence: { type: "speech", segments: translated },
      });
      return outputs;
    });

    const assembled = await runPhase("assemble", async () => {
      const result = await mediaEngine.assemble(
        inputName,
        state.file,
        speechSegments,
        state.metadata.duration,
        (progress) => {
          updatePhase("assemble", {
            progress: clampProgress(progress),
            evidence: { type: "progress", title: "Assembling local output", text: "The original video stream is copied where the source codec permits it." },
          });
        },
      );
      updatePhase("assemble", {
        metrics: [
          ["Duration", formatDuration(state.metadata.duration)],
          ["Container", result.mimeType],
          ["Video", state.file.type.startsWith("video/") ? "Stream copied" : "Audio input"],
          ["Upload", "None"],
        ],
        evidence: { type: "progress", title: "Assembly complete", text: `${formatBytes(result.blob.size)} generated locally.` },
      });
      return result;
    });

    await runPhase("deliver", async () => {
      createArtifacts(assembled);
      const cost = estimateCost();
      updatePhase("deliver", {
        metrics: [
          ["Artifacts", String(state.artifacts.length)],
          ["API cost", `${usd(cost.total)} · ${inr(cost.total)}`],
          ["Media upload", "0 bytes"],
          ["Persistence", "Until reload"],
        ],
        evidence: { type: "artifacts", artifacts: state.artifacts },
      });
    });
    selectPhase("deliver");
  } catch (error) {
    if (error.name !== "AbortError") {
      const phase = state.currentPhase ?? "inspect";
      updatePhase(phase, { status: "error", evidence: { type: "error", text: safeError(error) } });
      addActivity(phase, safeError(error));
      showError(safeError(error));
      selectPhase(phase);
    }
  } finally {
    state.running = false;
    state.currentPhase = null;
    state.abortController = null;
    elements["cancel-button"].hidden = true;
    stopTimer();
    if (usedGemini) forgetKey();
    validateReadyState();
  }
}

async function runPhase(id, operation) {
  ensureNotCancelled();
  state.currentPhase = id;
  const started = performance.now();
  updatePhase(id, { status: "running", progress: 2, evidence: { type: "progress", title: phaseDefinition(id)[1], text: "Starting…" } });
  selectPhase(id);
  addActivity(id, `${phaseDefinition(id)[1]} started`);
  const result = await operation();
  ensureNotCancelled();
  const seconds = (performance.now() - started) / 1000;
  updatePhase(id, { status: "complete", progress: 100, seconds });
  addActivity(id, `${phaseDefinition(id)[1]} complete in ${formatSeconds(seconds)}`);
  return result;
}

function handleModelEvent(message) {
  const phase = { transcription: "transcribe", translation: "translate", speech: "speech" }[message.task];
  if (!phase || state.currentPhase !== phase) return;
  if (message.status === "model-progress") {
    const progress = message.progress == null ? state.phases[phase].progress : Math.min(82, 5 + message.progress * .75);
    updatePhase(phase, {
      progress,
      evidence: {
        type: "progress",
        title: `Downloading ${shortFile(message.file)}`,
        text: message.total ? `${formatBytes(message.loaded ?? 0)} of ${formatBytes(message.total)}` : "Model data is cached by the browser for later runs.",
      },
    });
  } else if (message.status === "loading" || message.status === "running") {
    updatePhase(phase, { evidence: { type: "progress", title: message.message, text: "No project media leaves this tab." } });
  } else if (message.status === "stream") {
    updatePhase(phase, { progress: Math.min(94, state.phases[phase].progress + 1), evidence: { type: "stream", text: message.text } });
  } else if (message.status === "segment") {
    const isTranslation = phase === "translate";
    updatePhase(phase, {
      progress: 8 + Math.round((message.index / message.total) * 84),
      evidence: {
        type: "progress",
        title: message.message,
        text: isTranslation ? message.text : "MMS Hindi is running through local WASM.",
      },
    });
  }
}

function normalizeTranscript(result, duration) {
  const value = Array.isArray(result) ? result[0] : result;
  const chunks = value?.chunks ?? [];
  if (!chunks.length && value?.text?.trim()) {
    return [{ start: 0, end: duration, original: value.text.trim() }];
  }
  return chunks
    .map((chunk, index) => {
      const start = Number(chunk.timestamp?.[0] ?? (index * 15));
      const end = Number(chunk.timestamp?.[1] ?? Math.min(duration, start + 15));
      return { start, end: Math.max(start + .2, Math.min(duration, end)), original: String(chunk.text ?? "").trim() };
    })
    .filter((segment) => segment.original)
    .slice(0, 12);
}

function translationPrompt(segments) {
  const payload = segments.map((segment, id) => ({
    id,
    start: round(segment.start),
    end: round(segment.end),
    duration: round(segment.end - segment.start),
    max_words: Math.max(4, Math.floor((segment.end - segment.start) * 2.4)),
    text: segment.original,
  }));
  return `Translate these English lecture segments into natural spoken Hindi in Devanagari.
Preserve technical names, symbols and familiar English classroom vocabulary.
Do not exceed max_words. Return only a JSON array of objects with keys id and translated.

${JSON.stringify(payload)}`;
}

function parseTranslation(raw, originals) {
  const cleaned = String(raw).replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/```(?:json)?|```/gi, "").trim();
  const match = cleaned.match(/\[[\s\S]*\]/);
  if (!match) throw new Error("The translation model did not return a JSON array. Try Gemini or the larger local model.");
  let parsed;
  try {
    parsed = JSON.parse(match[0]);
  } catch {
    throw new Error("The translation JSON was invalid. Try the run again or select Gemini.");
  }
  const byId = new Map(parsed.map((item) => [Number(item.id), String(item.translated ?? "").trim()]));
  return originals.map((segment, id) => {
    const translated = byId.get(id);
    if (!translated) throw new Error(`Translation is missing segment ${id + 1}.`);
    return { ...segment, translated };
  });
}

function addGeminiUsage(kind, usage) {
  const input = Number(usage.promptTokenCount ?? 0);
  const output = Number(usage.candidatesTokenCount ?? 0);
  if (kind === "translation") {
    state.usage.translationInput += input;
    state.usage.translationOutput += output;
  } else {
    state.usage.ttsInput += input;
    state.usage.ttsOutput += output;
  }
}

function estimateCost() {
  const translation = state.usage.translationInput / 1_000_000 * .30 + state.usage.translationOutput / 1_000_000 * 2.50;
  const tts = state.usage.ttsInput / 1_000_000 * .50 + state.usage.ttsOutput / 1_000_000 * 10;
  return { translation, tts, total: translation + tts };
}

function createArtifacts(assembled) {
  const base = state.file.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "_").slice(0, 60) || "vidvaani";
  const transcript = {
    created_at: new Date().toISOString(),
    source: { name: state.file.name, duration: state.metadata.duration, size: state.file.size },
    privacy: { media_uploaded: false, api_key_persisted: false },
    providers: {
      transcription: { provider: "local", model: elements["stt-model"].value },
      translation: { provider: provider("translation"), model: elements["translation-model"].value },
      speech: { provider: provider("speech"), model: elements["speech-model"].value },
    },
    usage: state.usage,
    estimated_cost: estimateCost(),
    segments: state.translatedSegments,
  };
  addArtifact(`${base}_hindi_browser.${assembled.mimeType === "video/webm" ? "webm" : assembled.mimeType === "video/mp4" ? "mp4" : "wav"}`, assembled.blob, "Hindi media");
  addArtifact(`${base}_hindi.srt`, new Blob([segmentsToSrt(state.translatedSegments)], { type: "application/x-subrip" }), "Hindi subtitles");
  addArtifact(`${base}_pipeline.json`, new Blob([JSON.stringify(transcript, null, 2)], { type: "application/json" }), "Run data");
  elements["outputs-section"].hidden = false;
  elements["output-video"].src = state.artifacts[0].url;
  elements["output-actions"].replaceChildren(...state.artifacts.map((artifact) => {
    const link = document.createElement("a");
    link.href = artifact.url;
    link.download = artifact.name;
    link.textContent = `Download ${artifact.label}`;
    return link;
  }));
  elements["outputs-section"].scrollIntoView({ behavior: "smooth", block: "start" });
}

function addArtifact(name, blob, label) {
  state.artifacts.push({ name, blob, label, url: URL.createObjectURL(blob) });
}

function segmentsToSrt(segments) {
  return segments.map((segment, index) => [
    index + 1,
    `${srtTime(segment.start)} --> ${srtTime(segment.end)}`,
    segment.translated,
    "",
  ].join("\n")).join("\n");
}

function srtTime(seconds) {
  const milliseconds = Math.round(seconds * 1000);
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor(milliseconds / 60_000) % 60;
  const secs = Math.floor(milliseconds / 1000) % 60;
  const ms = milliseconds % 1000;
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)},${String(ms).padStart(3, "0")}`;
}

function renderPhaseButtons() {
  elements["phase-list"].replaceChildren(...PHASE_DEFINITIONS.map(([id, name], index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `phase-button ${state.phases[id].status}${state.selectedPhase === id ? " selected" : ""}`;
    button.addEventListener("click", () => selectPhase(id));
    button.append(
      textElement("span", "phase-index", String(index + 1).padStart(2, "0")),
      textElement("span", "phase-name", name),
      textElement("span", "phase-state", statusLabel(state.phases[id].status)),
    );
    item.append(button);
    return item;
  }));
}

function selectPhase(id) {
  state.selectedPhase = id;
  renderPhaseButtons();
  renderDetail();
}

function renderDetail() {
  const [id, title, description] = phaseDefinition(state.selectedPhase);
  const phase = state.phases[id];
  const index = PHASE_DEFINITIONS.findIndex(([phaseId]) => phaseId === id) + 1;
  elements["detail-eyebrow"].textContent = `PHASE ${String(index).padStart(2, "0")} · ${statusLabel(phase.status).toUpperCase()}`;
  elements["detail-title"].textContent = title;
  elements["detail-description"].textContent = description;
  elements["detail-time"].textContent = phase.seconds == null ? "—" : formatSeconds(phase.seconds);
  elements["detail-progress"].style.width = `${phase.progress}%`;
  elements["detail-metrics"].replaceChildren(...(phase.metrics.length ? phase.metrics : [["State", statusLabel(phase.status)], ["Data location", "This browser"]]).map(([key, value]) => {
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    wrapper.append(dt, dd);
    return wrapper;
  }));
  renderEvidence(phase.evidence);
}

function renderEvidence(evidence) {
  const container = elements["detail-evidence"];
  container.replaceChildren();
  if (!evidence || evidence.type === "empty") {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.append(textElement("span", "", "NO LOCAL DATA YET"), textElement("p", "", "Intermediate files and text remain in this tab until reload."));
    container.append(empty);
  } else if (evidence.type === "file") {
    container.append(progressEvidence("Local file accepted", `${evidence.file.name} will be read through the browser File API; it is not uploaded.`));
  } else if (evidence.type === "progress") {
    container.append(progressEvidence(evidence.title, evidence.text));
  } else if (evidence.type === "stream") {
    container.append(progressEvidence("Local model output", evidence.text || "Generating…"));
  } else if (evidence.type === "transcript") {
    container.append(...evidence.segments.slice(0, 8).map((segment) => transcriptRow(segment)));
  } else if (evidence.type === "translation") {
    container.append(...evidence.segments.slice(0, 8).map((segment) => translationRow(segment)));
  } else if (evidence.type === "speech") {
    container.append(...evidence.segments.slice(0, 8).map((segment) => transcriptRow({ ...segment, original: segment.translated })));
  } else if (evidence.type === "artifacts") {
    container.append(progressEvidence("Artifacts ready", evidence.artifacts.map((artifact) => artifact.name).join(" · ")));
  } else if (evidence.type === "error") {
    container.append(progressEvidence("Phase stopped", evidence.text));
  }
}

function progressEvidence(title, text) {
  const wrapper = document.createElement("div");
  wrapper.className = "progress-evidence";
  wrapper.append(textElement("strong", "", title), textElement("p", "", text));
  return wrapper;
}

function transcriptRow(segment) {
  const row = document.createElement("div");
  row.className = "transcript-row";
  row.append(textElement("time", "timecode", `${shortTime(segment.start)}–${shortTime(segment.end)}`));
  const body = document.createElement("div");
  body.append(textElement("span", "evidence-label", "TRANSCRIPT"), textElement("p", "", segment.original));
  row.append(body);
  return row;
}

function translationRow(segment) {
  const row = document.createElement("div");
  row.className = "translation-row";
  row.append(textElement("time", "timecode", `${shortTime(segment.start)}–${shortTime(segment.end)}`));
  const english = document.createElement("div");
  english.append(textElement("span", "evidence-label", "ENGLISH"), textElement("p", "", segment.original));
  const hindi = document.createElement("div");
  hindi.className = "hindi";
  hindi.append(textElement("span", "evidence-label", "HINDI"), textElement("p", "", segment.translated));
  row.append(english, hindi);
  return row;
}

function updatePhase(id, patch) {
  state.phases[id] = { ...state.phases[id], ...patch };
  renderPhaseButtons();
  if (state.selectedPhase === id) renderDetail();
}

function resetPhases(preserveInspection) {
  const inspection = state.phases.inspect;
  state.phases = Object.fromEntries(PHASE_DEFINITIONS.map(([id]) => [id, freshPhase()]));
  if (preserveInspection && inspection.status === "complete") state.phases.inspect = inspection;
  state.segments = [];
  state.translatedSegments = [];
  renderPhaseButtons();
  renderDetail();
}

function renderActivity() {
  if (!state.activity.length) {
    const item = document.createElement("li");
    item.className = "activity-empty";
    item.textContent = "No processing events yet.";
    elements["activity-list"].replaceChildren(item);
    return;
  }
  elements["activity-list"].replaceChildren(...state.activity.map((event) => {
    const item = document.createElement("li");
    item.append(textElement("time", "", event.time), textElement("strong", "", event.phase), textElement("span", "", event.message));
    return item;
  }));
}

function addActivity(phase, message) {
  state.activity.unshift({
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    phase,
    message,
  });
  state.activity = state.activity.slice(0, 30);
  renderActivity();
}

function cancelRun() {
  state.cancelled = true;
  state.abortController?.abort();
  modelClient.reset();
  mediaEngine.reset();
  if (state.currentPhase) {
    updatePhase(state.currentPhase, { status: "error", evidence: { type: "error", text: "Run cancelled by the user." } });
    addActivity(state.currentPhase, "Run cancelled");
  }
}

function ensureNotCancelled() {
  if (state.cancelled) throw new DOMException("Run cancelled", "AbortError");
}

function forgetKey() {
  state.geminiKey = "";
  elements["gemini-key"].value = "";
  validateReadyState();
}

function cleanupSensitiveState() {
  forgetKey();
  clearArtifacts();
}

function clearArtifacts() {
  for (const artifact of state.artifacts) URL.revokeObjectURL(artifact.url);
  state.artifacts = [];
  elements["output-video"].removeAttribute("src");
  elements["output-video"].load();
  elements["output-actions"].replaceChildren();
  elements["outputs-section"].hidden = true;
}

function startTimer() {
  stopTimer();
  state.timer = window.setInterval(() => {
    elements["run-time"].textContent = formatClock((performance.now() - state.runStartedAt) / 1000);
  }, 500);
}

function stopTimer() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (state.runStartedAt) elements["run-time"].textContent = formatClock((performance.now() - state.runStartedAt) / 1000);
}

function phaseDefinition(id) {
  return PHASE_DEFINITIONS.find(([phaseId]) => phaseId === id);
}

function statusLabel(status) {
  return ({ waiting: "Waiting", running: "Running", complete: "Complete", error: "Stopped" })[status] ?? status;
}

function textElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function showError(message) { elements["form-error"].textContent = message; }
function clearError() { elements["form-error"].textContent = ""; }
function safeError(error) { return error instanceof Error ? error.message : "The browser operation failed."; }
function shortModel(value) { return value.split("/").pop().replace(/-ONNX$/i, ""); }
function shortFile(value) { return value.split("/").pop(); }
function round(value) { return Math.round(value * 100) / 100; }
function clampProgress(value) { return Math.max(4, Math.min(95, Math.round((Number(value) || 0) * 100))); }
function formatBytes(value) {
  if (!Number.isFinite(value)) return "Unknown";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}
function formatDuration(seconds) { return seconds >= 60 ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s` : `${seconds.toFixed(1)}s`; }
function formatSeconds(seconds) { return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`; }
function formatClock(seconds) { return `${pad(Math.floor(seconds / 60))}:${pad(Math.floor(seconds % 60))}`; }
function shortTime(seconds) { return `${pad(Math.floor(seconds / 60))}:${pad(Math.floor(seconds % 60))}`; }
function pad(value) { return String(value).padStart(2, "0"); }
function usd(value) { return `$${value.toFixed(value < .01 ? 5 : 3)}`; }
function inr(value) { return `₹${(value * USD_TO_INR).toFixed(2)}`; }

initialize();
