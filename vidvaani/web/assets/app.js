const PHASES = [
  { id: "download", label: "Source media", engine: "yt-dlp", weight: 0.12, summary: "Resolve the YouTube source, download the best stream up to 1080p, and extract transcription-ready audio." },
  { id: "analyze", label: "Intro analysis", engine: "ffmpeg", weight: 0.05, summary: "Scan the opening audio to find where lecture speech begins while preserving any intro music." },
  { id: "transcribe", label: "Speech to text", engine: "MLX Whisper", weight: 0.24, summary: "Transcribe English speech locally, retain timestamps, and group phrases into dubbing-sized segments." },
  { id: "translate", label: "Hindi translation", engine: "Gemini Flash", weight: 0.17, summary: "Translate each timed English segment into spoken Hindi while protecting technical vocabulary and the time budget." },
  { id: "synthesize", label: "Voice synthesis", engine: "TTS", weight: 0.27, summary: "Generate one Hindi speech clip per translated segment, then measure and fit it to the available speaking slot." },
  { id: "assemble", label: "Timing & mix", engine: "ffmpeg", weight: 0.13, summary: "Place Hindi clips on the original timeline, preserve non-speech audio, and stop demo output after the final processed segment." },
  { id: "deliver", label: "Deliverables", engine: "MP4 · SRT · JSON", weight: 0.02, summary: "Index the dubbed video, subtitles, aligned transcripts, timing breakdown, and final cost report." },
];

const FALLBACK_VOICES = {
  sarvam: ["abhilash", "anushka", "arya", "hitesh", "karun", "manisha", "vidya"],
  gemini: ["Aoede", "Charon", "Fenrir", "Iapetus", "Kore", "Orus", "Puck", "Sadaltager"],
  edge: ["male", "female"],
};

const FALLBACK_VOICE_PROFILES = {
  sarvam: {
    abhilash: { gender: "Male", tone: "Steady lecture tone", description: "Native Indian prosody; measured and dependable for technical explanations." },
    anushka: { gender: "Female", tone: "Warm and clear", description: "Native Indian prosody; expressive narration with crisp diction." },
    arya: { gender: "Female", tone: "Balanced", description: "Native Indian prosody; a neutral alternative for long-form lessons." },
    hitesh: { gender: "Male", tone: "Direct", description: "Native Indian prosody; a distinct male timbre for lecture material." },
    karun: { gender: "Male", tone: "Conversational", description: "Native Indian prosody; a softer alternative for explanatory teaching." },
    manisha: { gender: "Female", tone: "Composed", description: "Native Indian prosody; even delivery for sustained narration." },
    vidya: { gender: "Female", tone: "Clear lecture delivery", description: "Native Indian prosody; a classroom-friendly female voice." },
  },
  gemini: {
    Aoede: { gender: "Female", tone: "Breezy", description: "Google's breezy voice profile; light delivery for accessible narration." },
    Charon: { gender: "Male", tone: "Informative", description: "Google's informative profile; natural delivery with a slight Western accent in Hindi." },
    Fenrir: { gender: "Male", tone: "Excitable", description: "Google's energetic profile; more animated than a conventional lecture voice." },
    Iapetus: { gender: "Male", tone: "Clear", description: "Google's clear profile; precise diction and an even pace." },
    Kore: { gender: "Female", tone: "Firm", description: "Google's firm profile; confident, structured delivery." },
    Orus: { gender: "Male", tone: "Firm", description: "Google's firm profile; confident and well suited to lecture narration." },
    Puck: { gender: "Male", tone: "Upbeat", description: "Google's upbeat profile; lively delivery for shorter material." },
    Sadaltager: { gender: "Male", tone: "Knowledgeable", description: "Google's knowledgeable profile; warm, authoritative delivery." },
  },
  edge: {
    male: { gender: "Male", tone: "Madhur", description: "Microsoft hi-IN-MadhurNeural; free Hindi fallback with a male voice." },
    female: { gender: "Female", tone: "Swara", description: "Microsoft hi-IN-SwaraNeural; free Hindi fallback with a female voice." },
  },
};

const elements = {
  form: document.querySelector("#job-form"),
  url: document.querySelector("#source-url"),
  urlError: document.querySelector("#url-error"),
  submitError: document.querySelector("#submit-error"),
  runButton: document.querySelector("#run-button"),
  sampleRun: document.querySelector("#sample-run"),
  voice: document.querySelector("#voice-select"),
  voiceProfileMeta: document.querySelector("#voice-profile-meta"),
  voiceProfileDescription: document.querySelector("#voice-profile-description"),
  phaseList: document.querySelector("#phase-list"),
  phaseInspectorEyebrow: document.querySelector("#phase-inspector-eyebrow"),
  phaseInspectorTitle: document.querySelector("#phase-inspector-title"),
  phaseInspectorSummary: document.querySelector("#phase-inspector-summary"),
  inspectorProgressValue: document.querySelector("#inspector-progress-value"),
  inspectorProgressBar: document.querySelector("#inspector-progress-bar"),
  phaseStats: document.querySelector("#phase-stats"),
  phaseEvidence: document.querySelector("#phase-evidence"),
  followPhase: document.querySelector("#follow-phase"),
  progressValue: document.querySelector("#progress-value"),
  progressBar: document.querySelector("#progress-bar"),
  phasePosition: document.querySelector("#phase-position"),
  elapsedTime: document.querySelector("#elapsed-time"),
  costUsd: document.querySelector("#cost-usd"),
  costInr: document.querySelector("#cost-inr"),
  runState: document.querySelector("#run-state-label"),
  runTitle: document.querySelector("#run-title"),
  runSubtitle: document.querySelector("#run-subtitle"),
  liveLabel: document.querySelector("#live-label"),
  liveCaption: document.querySelector(".live-caption"),
  eventLog: document.querySelector("#event-log"),
  activityEmpty: document.querySelector("#activity-empty"),
  clearLog: document.querySelector("#clear-log"),
  outputEmpty: document.querySelector("#output-empty"),
  outputContent: document.querySelector("#output-content"),
  resultVideo: document.querySelector("#result-video"),
  artifactList: document.querySelector("#artifact-list"),
  artifactCount: document.querySelector("#artifact-count"),
  resultStats: document.querySelector("#result-stats"),
  transcriptPanel: document.querySelector("#transcript-panel"),
  transcriptLines: document.querySelector("#transcript-lines"),
  historyList: document.querySelector("#history-list"),
  historyCount: document.querySelector("#history-count"),
  systemPill: document.querySelector("#system-pill"),
  systemLabel: document.querySelector("#system-label"),
  systemPopover: document.querySelector("#system-popover"),
  closeSystem: document.querySelector("#close-system"),
  serviceList: document.querySelector("#service-list"),
  toastRegion: document.querySelector("#toast-region"),
};

const initialPhases = () => PHASES.map((phase) => ({
  ...phase,
  status: "pending",
  progress: 0,
  message: "Waiting",
  started_at: null,
  finished_at: null,
  details: {},
}));

const state = {
  job: null,
  phases: initialPhases(),
  events: [],
  eventSource: null,
  voices: FALLBACK_VOICES,
  voiceProfiles: FALLBACK_VOICE_PROFILES,
  transcriptCache: {},
  activeLanguage: "english",
  activePhaseId: "download",
  phasePinned: false,
};

function escapeText(value) {
  return String(value ?? "");
}

function formatDuration(seconds, compact = false) {
  if (!Number.isFinite(seconds) || seconds < 0) return compact ? "—" : "00:00";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;
  if (hours) return `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function clockTime(timestamp) {
  if (!timestamp) return "--:--:--";
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour12: false });
}

function titleCase(value) {
  return escapeText(value).replace(/\b\w/g, (character) => character.toUpperCase());
}

function phaseDuration(phase) {
  if (Number.isFinite(phase.details?.duration_seconds)) return phase.details.duration_seconds;
  if (!phase.started_at) return null;
  return (phase.finished_at || Date.now() / 1000) - phase.started_at;
}

function initializePhaseRows() {
  elements.phaseList.replaceChildren();
  PHASES.forEach((phase, index) => {
    const item = document.createElement("li");
    item.className = "phase-item pending";
    item.dataset.phaseId = phase.id;
    item.style.setProperty("--index", index);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "phase-button";
    button.setAttribute("aria-label", `Inspect ${phase.label}`);

    const indexNode = document.createElement("span");
    indexNode.className = "phase-index";
    indexNode.textContent = String(index + 1).padStart(2, "0");

    const title = document.createElement("span");
    title.className = "phase-title";
    const strong = document.createElement("strong");
    strong.textContent = phase.label;
    const engine = document.createElement("small");
    engine.textContent = phase.engine;
    title.append(strong, engine);

    const message = document.createElement("span");
    message.className = "phase-message";
    message.textContent = "Waiting";

    const time = document.createElement("span");
    time.className = "phase-time";
    time.textContent = "—";

    const markerWrap = document.createElement("span");
    markerWrap.className = "phase-state";
    const marker = document.createElement("i");
    marker.className = "state-marker";
    marker.setAttribute("aria-label", "pending");
    markerWrap.append(marker);

    button.append(indexNode, title, message, time, markerWrap);
    button.addEventListener("click", () => selectPhase(phase.id, true));
    item.append(button);
    elements.phaseList.append(item);
  });
}

function renderPhases() {
  if (elements.phaseList.children.length !== PHASES.length) initializePhaseRows();

  state.phases.forEach((phase) => {
    const item = elements.phaseList.querySelector(`[data-phase-id="${phase.id}"]`);
    if (!item) return;
    const selected = phase.id === state.activePhaseId;
    item.className = `phase-item ${phase.status}${selected ? " selected" : ""}`;
    item.style.setProperty("--phase-progress", `${Math.round((phase.progress || 0) * 100)}%`);
    item.querySelector(".phase-message").textContent = phase.message;
    const duration = phaseDuration(phase);
    item.querySelector(".phase-time").textContent = duration === null
      ? "—"
      : `${duration.toFixed(duration < 10 ? 1 : 0)}s`;
    item.querySelector(".state-marker").setAttribute("aria-label", phase.status);
    item.querySelector(".phase-button").setAttribute("aria-pressed", String(selected));
  });
  renderMeter();
  renderPhaseInspector();
}

function renderMeter() {
  const overall = state.phases.reduce((sum, phase) => sum + phase.weight * phase.progress, 0);
  const percentage = Math.min(100, Math.round(overall * 100));
  const complete = state.phases.filter((phase) => ["complete", "cached"].includes(phase.status)).length;
  elements.progressValue.textContent = percentage;
  elements.progressBar.style.width = `${percentage}%`;
  elements.phasePosition.textContent = `${complete} / ${PHASES.length} phases`;
  const costs = state.job?.costs || { total_cost_usd: 0, total_cost_inr: 0 };
  elements.costUsd.textContent = `$${Number(costs.total_cost_usd || 0).toFixed(4)}`;
  elements.costInr.textContent = `₹${Number(costs.total_cost_inr || 0).toFixed(2)}`;
}

function formatPhaseSeconds(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  return formatDuration(seconds, true);
}

function formatResolution(details) {
  return details?.width && details?.height ? `${details.width} × ${details.height}` : "—";
}

function selectedPhaseStats(phase) {
  const details = phase.details || {};
  const duration = phaseDuration(phase);
  const costs = details.costs || state.job?.costs || {};
  const rows = [
    ["State", titleCase(phase.status)],
    ["Time taken", formatPhaseSeconds(duration)],
  ];
  const add = (label, value, wrap = false) => {
    if (value !== undefined && value !== null && value !== "") rows.push([label, String(value), wrap]);
  };

  if (phase.id === "download") {
    add("Video duration", Number.isFinite(details.duration) ? formatDuration(details.duration, true) : "—");
    add("Source resolution", formatResolution(details));
    if (details.bytes_total) add("Downloaded", `${formatBytes(details.bytes_downloaded || 0)} / ${formatBytes(details.bytes_total)}`);
    add("Source title", details.title, true);
  } else if (phase.id === "analyze") {
    add("Speech begins", Number.isFinite(details.intro_offset) ? `${details.intro_offset.toFixed(1)} s` : "—");
    add("Method", "Silence boundary scan");
  } else if (phase.id === "transcribe") {
    const completed = details.seconds_complete;
    const total = details.seconds_total || details.audio_duration;
    add("Audio scanned", Number.isFinite(completed) && Number.isFinite(total) ? `${formatDuration(completed, true)} / ${formatDuration(total, true)}` : Number.isFinite(total) ? formatDuration(total, true) : "—");
    add("Segments", details.segments ?? "—");
    add("Language", details.source_language?.toUpperCase() || "English");
    add("Model", details.model?.split("/").pop() || "MLX Whisper", true);
  } else if (phase.id === "translate") {
    const done = details.segments_complete ?? details.segments;
    const total = details.segments_total ?? details.segments;
    add("Segments", Number.isFinite(done) && Number.isFinite(total) ? `${done} / ${total}` : "—");
    add("Model", details.model || "gemini-2.5-flash", true);
    add("Input tokens", costs.translation?.input_tokens?.toLocaleString() ?? "—");
    add("Output tokens", costs.translation?.output_tokens?.toLocaleString() ?? "—");
    add("Translation cost", Number.isFinite(costs.translation?.cost_usd) ? `$${costs.translation.cost_usd.toFixed(6)}` : "$0.000000");
  } else if (phase.id === "synthesize") {
    const done = details.segments_complete ?? details.segments;
    const total = details.segments_total ?? details.segments;
    const ttsCost = costs.tts_sarvam || costs.tts_gemini;
    add("Speech clips", Number.isFinite(done) && Number.isFinite(total) ? `${done} / ${total}` : "—");
    add("Backend", titleCase(details.backend || state.job?.request?.backend || "TTS"));
    add("Voice", titleCase(details.voice || state.job?.request?.voice || "—"));
    add("Cached clips", details.cached_segments ?? "—");
    add("API calls", ttsCost?.calls ?? "—");
    add("TTS cost", Number.isFinite(ttsCost?.cost_usd) ? `$${ttsCost.cost_usd.toFixed(6)}` : "$0.000000");
  } else if (phase.id === "assemble") {
    add("Output duration", Number.isFinite(details.duration) ? formatDuration(details.duration, true) : "—");
    add("Source duration", Number.isFinite(details.source_duration) ? formatDuration(details.source_duration, true) : "—");
    add("Output resolution", formatResolution(details));
    add("Speech clips", details.segments_used ?? "—");
    add("Demo boundary", details.demo_trimmed === true ? "Ends after final processed segment" : details.demo_trimmed === false ? "Full source" : "—", true);
  } else if (phase.id === "deliver") {
    const result = state.job?.result || {};
    add("Files", details.artifact_count ?? state.job?.artifacts?.length ?? "—");
    add("Video duration", Number.isFinite(result.duration) ? formatDuration(result.duration, true) : "—");
    add("Segments", result.segments ?? "—");
    add("Total runtime", Number.isFinite(result.timings?.total) ? formatDuration(result.timings.total, true) : "—");
  }
  return rows;
}

function appendEvidenceHeading(container, title, meta) {
  const heading = document.createElement("div");
  heading.className = "evidence-heading";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const copy = document.createElement("span");
  copy.textContent = meta;
  heading.append(strong, copy);
  container.append(heading);
}

function renderTextPreview(container, preview, language = "English") {
  appendEvidenceHeading(container, `${language} segment preview`, `${preview.length} shown`);
  const list = document.createElement("div");
  list.className = "text-preview";
  preview.forEach((segment) => {
    const row = document.createElement("div");
    row.className = "text-preview-row";
    const time = document.createElement("time");
    time.textContent = `${formatDuration(segment.start, true)}–${formatDuration(segment.end, true)}`;
    const copy = document.createElement("p");
    copy.textContent = segment.text || segment.translated || "";
    row.append(time, copy);
    list.append(row);
  });
  container.append(list);
}

function renderTranslationPreview(container, preview) {
  appendEvidenceHeading(container, "Aligned translation", `${preview.length} segment${preview.length === 1 ? "" : "s"} shown`);
  const list = document.createElement("div");
  list.className = "translation-preview";
  preview.forEach((segment) => {
    const row = document.createElement("div");
    row.className = "translation-row";
    const time = document.createElement("time");
    time.textContent = `${formatDuration(segment.start, true)}–${formatDuration(segment.end, true)}`;
    const english = document.createElement("div");
    english.className = "translation-copy";
    const englishLabel = document.createElement("span");
    englishLabel.textContent = "ENGLISH";
    const englishCopy = document.createElement("p");
    englishCopy.textContent = segment.original || "";
    english.append(englishLabel, englishCopy);
    const hindi = document.createElement("div");
    hindi.className = "translation-copy hindi";
    const hindiLabel = document.createElement("span");
    hindiLabel.textContent = "HINDI";
    const hindiCopy = document.createElement("p");
    hindiCopy.lang = "hi";
    hindiCopy.textContent = segment.translated || segment.text || "";
    hindi.append(hindiLabel, hindiCopy);
    row.append(time, english, hindi);
    list.append(row);
  });
  container.append(list);
}

function renderTechnicalNote(container, title, copy) {
  const note = document.createElement("div");
  note.className = "technical-note";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const paragraph = document.createElement("p");
  paragraph.textContent = copy;
  note.append(strong, paragraph);
  container.append(note);
}

function renderPhaseEvidence(phase) {
  elements.phaseEvidence.replaceChildren();
  const details = phase.details || {};
  const preview = Array.isArray(details.preview) ? details.preview : [];

  if (phase.id === "translate" && preview.length) {
    renderTranslationPreview(elements.phaseEvidence, preview);
    return;
  }
  if (["transcribe", "synthesize"].includes(phase.id) && preview.length) {
    renderTextPreview(elements.phaseEvidence, preview, phase.id === "transcribe" ? "English" : "Hindi");
    return;
  }
  if (phase.id === "download" && details.title) {
    renderTechnicalNote(elements.phaseEvidence, details.title, `${formatDuration(details.duration, true)} source at ${formatResolution(details)}. The video stream is copied into the final MP4 without a resolution-reducing re-encode.`);
    return;
  }
  if (phase.id === "analyze" && Number.isFinite(details.intro_offset)) {
    renderTechnicalNote(elements.phaseEvidence, "Opening boundary", details.intro_offset > 0 ? `Original audio is preserved until ${details.intro_offset.toFixed(1)} seconds, where lecture speech begins.` : "No separate intro boundary was detected; dubbing begins at the first speech segment.");
    return;
  }
  if (phase.id === "assemble" && phase.status !== "pending") {
    renderTechnicalNote(elements.phaseEvidence, details.demo_trimmed ? "No untranslated tail" : "Full-length assembly", details.demo_trimmed ? "This demo MP4 stops after the final requested speech segment, so the remaining English lecture is not appended." : "The full source timeline is being assembled because full-video processing was requested.");
    return;
  }
  if (phase.id === "deliver" && state.job?.artifacts?.length) {
    renderTechnicalNote(elements.phaseEvidence, "Artifacts indexed", `${state.job.artifacts.length} files are available in Output: the MP4, SRT subtitles, and aligned English/Hindi JSON transcripts.`);
    return;
  }

  const empty = document.createElement("div");
  empty.className = "evidence-empty";
  const label = document.createElement("span");
  label.textContent = phase.status === "pending" ? "WAITING FOR PHASE" : "LIVE EVIDENCE";
  const copy = document.createElement("p");
  copy.textContent = phase.status === "pending"
    ? "Select any phase now; its measurements and evidence will populate here as the run advances."
    : phase.message;
  empty.append(label, copy);
  elements.phaseEvidence.append(empty);
}

function renderPhaseInspector() {
  const phase = state.phases.find((item) => item.id === state.activePhaseId) || state.phases[0];
  if (!phase) return;
  const definition = PHASES.find((item) => item.id === phase.id) || phase;
  const index = PHASES.findIndex((item) => item.id === phase.id);
  elements.phaseInspectorEyebrow.textContent = `PHASE ${String(index + 1).padStart(2, "0")} · ${phase.status.toUpperCase()}`;
  elements.phaseInspectorTitle.textContent = phase.label;
  elements.phaseInspectorSummary.textContent = definition.summary;
  const percentage = Math.round((phase.progress || 0) * 100);
  elements.inspectorProgressValue.textContent = `${percentage}%`;
  elements.inspectorProgressBar.style.width = `${percentage}%`;
  elements.followPhase.classList.toggle("active", !state.phasePinned);
  elements.followPhase.setAttribute("aria-pressed", String(!state.phasePinned));

  elements.phaseStats.replaceChildren();
  selectedPhaseStats(phase).forEach(([label, value, wrap]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    if (wrap) description.classList.add("wrap");
    row.append(term, description);
    elements.phaseStats.append(row);
  });
  renderPhaseEvidence(phase);
}

function selectPhase(phaseId, pin = false) {
  if (!state.phases.some((phase) => phase.id === phaseId)) return;
  state.activePhaseId = phaseId;
  state.phasePinned = pin;
  renderPhases();
}

function buildSampleSnapshot() {
  const preview = [
    {
      start: 0.0,
      end: 14.78,
      original: "Today I want to show the usage of a tool called GeoGebra.",
      translated: "आज मैं जिओजेब्रा नामक एक टूल का उपयोग दिखाना चाहता हूँ।",
    },
    {
      start: 14.78,
      end: 28.48,
      original: "We are given this uniform joint PDF in two dimensions.",
      translated: "हमें दो आयामों में यह यूनिफॉर्म जॉइंट पीडीएफ दिया गया है।",
    },
    {
      start: 28.48,
      end: 39.52,
      original: "The x range is between zero to two and the y range is between zero to two.",
      translated: "x और y की रेंज शून्य से दो तक है।",
    },
  ];
  const costs = {
    translation: { input_tokens: 815, output_tokens: 818, cost_usd: 0.002289 },
    tts_sarvam: { characters: 655, calls: 5, cost_usd: 0.010299, cost_inr: 0.98 },
    total_cost_usd: 0.012588,
    total_cost_inr: 1.2,
  };
  const now = Date.now() / 1000;
  let cursor = now - 60.4;
  const phaseData = {
    download: { duration: 2.5, details: { title: "2D integration, GeoGebra, joint distributions", duration: 684.5, width: 1864, height: 1080 } },
    analyze: { duration: 0.8, details: { intro_offset: 0.0 } },
    transcribe: { duration: 12.0, details: { audio_duration: 684.5, seconds_complete: 684, seconds_total: 684, segments: 5, source_language: "en", model: "mlx-community/distil-whisper-large-v3", preview: preview.map(({ start, end, original }) => ({ start, end, text: original })) } },
    translate: { duration: 17.0, details: { segments: 5, segments_complete: 5, segments_total: 5, model: "gemini-2.5-flash", preview, costs } },
    synthesize: { duration: 3.4, details: { segments: 5, segments_complete: 5, segments_total: 5, backend: "sarvam", voice: "abhilash", cached_segments: 0, preview: preview.map(({ start, end, translated }) => ({ start, end, text: translated })), costs } },
    assemble: { duration: 24.6, details: { duration: 58.464, source_duration: 684.5, width: 1864, height: 1080, segments_used: 5, demo_trimmed: true, output_name: "UuoVhUqWAFc_hindi_abhilash.mp4" } },
    deliver: { duration: 0.1, details: { artifact_count: 1, artifact_types: ["video"] } },
  };
  const phases = PHASES.map((phase) => {
    const sample = phaseData[phase.id];
    const startedAt = cursor;
    cursor += sample.duration;
    return {
      ...phase,
      status: "complete",
      progress: 1,
      message: phase.id === "deliver" ? "Ready — sample video available" : `${phase.label} complete`,
      started_at: startedAt,
      finished_at: cursor,
      details: { ...sample.details, duration_seconds: sample.duration },
    };
  });
  return {
    id: "sample",
    status: "complete",
    created_at: now - 61,
    started_at: now - 60.4,
    finished_at: now,
    current_phase: "deliver",
    progress: 1,
    error: null,
    request: { url: "https://www.youtube.com/watch?v=UuoVhUqWAFc", backend: "sarvam", voice: "abhilash", max_segments: 5 },
    phases,
    result: { title: "2D integration — five-segment Hindi sample", duration: 58.464, segments: 5, timings: { total: 60.4 }, costs },
    costs,
    artifacts: [{ key: "video", name: "geogebra_hindi_abhilash.mp4", size: 2990000, url: "/demo-assets/mini_demo/sarvam_abhilash.mp4" }],
  };
}

function loadSampleRun() {
  if (state.eventSource) state.eventSource.close();
  const sample = buildSampleSnapshot();
  state.events = sample.phases.map((phase) => ({
    phase: phase.id,
    status: "complete",
    message: phase.message,
    timestamp: phase.finished_at,
    details: phase.details,
  }));
  state.transcriptCache = {};
  state.activePhaseId = "translate";
  state.phasePinned = true;
  renderEvents();
  renderSnapshot(sample);
  showToast("Loaded a measured five-segment sample — no API calls");
}

function renderJobHeader() {
  if (!state.job) return;
  const job = state.job;
  const status = job.status;
  elements.liveCaption.className = `live-caption ${status}`;
  elements.liveLabel.textContent = status.toUpperCase();

  if (status === "queued") {
    elements.runState.textContent = "WAITING FOR PROCESSOR";
    elements.runTitle.textContent = "Run queued.";
    elements.runSubtitle.textContent = "The local worker will start this run when the current slot is available.";
  } else if (status === "running") {
    const current = state.phases.find((phase) => phase.id === job.current_phase);
    elements.runState.textContent = `RUN ${job.id.toUpperCase()}`;
    elements.runTitle.textContent = current ? current.label : "Processing source.";
    elements.runSubtitle.textContent = current?.message || "The pipeline is reporting live phase events.";
  } else if (status === "complete") {
    elements.runState.textContent = `RUN ${job.id.toUpperCase()} · COMPLETE`;
    elements.runTitle.textContent = job.result?.title || "Hindi dub ready.";
    elements.runSubtitle.textContent = `${job.result?.segments || 0} speech segments aligned across ${formatDuration(job.result?.duration)} of video.`;
  } else if (status === "failed") {
    elements.runState.textContent = `RUN ${job.id.toUpperCase()} · NEEDS ATTENTION`;
    elements.runTitle.textContent = "The run stopped at a visible boundary.";
    elements.runSubtitle.textContent = job.error || "Review the activity log, adjust the input, and retry.";
  }
}

function applyEvent(event) {
  const phase = state.phases.find((item) => item.id === event.phase);
  if (phase) {
    if (!phase.started_at && ["running", "complete", "cached"].includes(event.status)) {
      phase.started_at = event.timestamp;
    }
    phase.status = event.status;
    phase.progress = Number(event.progress ?? phase.progress);
    phase.message = event.message || phase.message;
    phase.details = { ...phase.details, ...(event.details || {}) };
    if (["complete", "cached", "failed"].includes(event.status)) {
      phase.finished_at = event.timestamp;
      if (event.status !== "failed") phase.progress = 1;
    }
  }
  if (state.job) {
    if (event.phase === "pipeline" && event.status === "running") {
      state.job.status = "running";
      state.job.started_at ||= event.timestamp;
    }
    if (event.status === "running" && phase) state.job.current_phase = event.phase;
    if (event.details?.costs) state.job.costs = event.details.costs;
  }
  if (phase && event.status === "running" && !state.phasePinned) {
    state.activePhaseId = event.phase;
  }

  const last = state.events[state.events.length - 1];
  if (last && last.phase === event.phase && last.status === "running" && event.status === "running") {
    state.events[state.events.length - 1] = event;
  } else if (event.phase !== "pipeline" || event.status !== "complete") {
    state.events.push(event);
  }
  state.events = state.events.slice(-120);
  renderPhases();
  renderJobHeader();
  renderEvents();
}

function renderEvents() {
  const hasEvents = state.events.length > 0;
  elements.activityEmpty.hidden = hasEvents;
  elements.eventLog.hidden = !hasEvents;
  if (!hasEvents) return;

  elements.eventLog.replaceChildren();
  state.events.forEach((event) => {
    const item = document.createElement("li");
    item.className = `event-item ${event.status}`;

    const time = document.createElement("time");
    time.className = "event-time";
    time.textContent = clockTime(event.timestamp);

    const phase = document.createElement("span");
    phase.className = "event-phase";
    phase.textContent = event.phase;

    const message = document.createElement("span");
    message.className = "event-message";
    message.textContent = event.message;

    item.append(time, phase, message);
    elements.eventLog.append(item);
  });
  elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
}

function renderOutput() {
  const job = state.job;
  const artifacts = job?.artifacts || [];
  const ready = job?.status === "complete" && artifacts.length;
  elements.outputEmpty.hidden = Boolean(ready);
  elements.outputContent.hidden = !ready;
  elements.artifactCount.textContent = ready ? `${artifacts.length} file${artifacts.length === 1 ? "" : "s"}` : "No files";
  if (!ready) return;

  const video = artifacts.find((artifact) => artifact.key === "video");
  if (video && elements.resultVideo.src !== new URL(video.url, location.href).href) {
    elements.resultVideo.src = video.url;
  }

  const labels = { video: "Video", subtitles: "SRT", english: "EN", hindi: "HI" };
  elements.artifactList.replaceChildren();
  artifacts.forEach((artifact) => {
    const link = document.createElement("a");
    link.className = "artifact-row";
    link.href = artifact.url;
    link.download = artifact.name;

    const kind = document.createElement("span");
    kind.className = "artifact-kind";
    kind.textContent = labels[artifact.key] || "File";

    const copy = document.createElement("span");
    copy.className = "artifact-copy";
    const name = document.createElement("strong");
    name.textContent = artifact.name;
    const size = document.createElement("small");
    size.textContent = formatBytes(artifact.size);
    copy.append(name, size);

    const arrow = document.createElement("span");
    arrow.className = "artifact-arrow";
    arrow.textContent = "→";
    arrow.setAttribute("aria-hidden", "true");
    link.append(kind, copy, arrow);
    elements.artifactList.append(link);
  });

  const result = job.result || {};
  const elapsed = job.started_at && job.finished_at ? job.finished_at - job.started_at : null;
  const stats = [
    ["Duration", formatDuration(result.duration, true)],
    ["Segments", String(result.segments ?? "—")],
    ["Process time", formatDuration(elapsed, true)],
  ];
  elements.resultStats.replaceChildren();
  stats.forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    wrapper.append(term, description);
    elements.resultStats.append(wrapper);
  });

  const hasTranscripts = artifacts.some((artifact) => ["english", "hindi"].includes(artifact.key));
  elements.transcriptPanel.hidden = !hasTranscripts;
  if (hasTranscripts) loadTranscript(state.activeLanguage);
}

async function loadTranscript(language) {
  if (!state.job || state.job.status !== "complete") return;
  const available = state.job.artifacts.some((artifact) => artifact.key === language);
  if (!available) return;

  state.activeLanguage = language;
  document.querySelectorAll("[data-language]").forEach((button) => {
    const active = button.dataset.language === language;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  elements.transcriptLines.innerHTML = '<div class="transcript-loading"><i class="skeleton"></i><i class="skeleton"></i></div>';

  try {
    if (!state.transcriptCache[language]) {
      const response = await fetch(`/api/jobs/${state.job.id}/transcript/${language}`);
      if (!response.ok) throw new Error("Transcript preview is not available");
      state.transcriptCache[language] = await response.json();
    }
    renderTranscript(language, state.transcriptCache[language]);
  } catch (error) {
    elements.transcriptLines.textContent = error.message;
  }
}

function renderTranscript(language, data) {
  elements.transcriptLines.replaceChildren();
  const segments = (data.segments || []).slice(0, 120);
  segments.forEach((segment) => {
    const row = document.createElement("div");
    row.className = "transcript-line";
    const timing = document.createElement("time");
    timing.textContent = `${formatDuration(segment.start)} — ${formatDuration(segment.end)}`;
    const copy = document.createElement("p");
    if (language === "hindi" && segment.original) {
      const original = document.createElement("strong");
      original.textContent = segment.original;
      copy.append(original, document.createTextNode(segment.text || ""));
    } else {
      copy.textContent = segment.text || "";
    }
    row.append(timing, copy);
    elements.transcriptLines.append(row);
  });
}

function renderSnapshot(snapshot) {
  state.job = snapshot;
  state.phases = snapshot.phases.map((phase) => ({ ...phase }));
  if (!state.phasePinned && snapshot.current_phase) state.activePhaseId = snapshot.current_phase;
  renderPhases();
  renderJobHeader();
  renderOutput();
  if (snapshot.status === "failed") showToast(snapshot.error || "Processing failed", "error");
  loadHistory();
}

function connectToJob(jobId) {
  if (state.eventSource) state.eventSource.close();
  const stream = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = stream;

  stream.addEventListener("progress", (message) => {
    applyEvent(JSON.parse(message.data));
  });
  stream.addEventListener("snapshot", (message) => {
    renderSnapshot(JSON.parse(message.data));
    stream.close();
  });
  stream.onerror = () => {
    if (state.job && !["complete", "failed"].includes(state.job.status)) {
      refreshJob(jobId);
    }
  };
}

async function refreshJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) return;
    renderSnapshot(await response.json());
  } catch (_) {
    // EventSource will reconnect automatically; keep the current visible state.
  }
}

function validateUrl() {
  elements.urlError.textContent = "";
  try {
    const parsed = new URL(elements.url.value.trim());
    const allowed = ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"];
    if (!allowed.includes(parsed.hostname)) throw new Error();
    return true;
  } catch (_) {
    elements.urlError.textContent = "Enter a valid youtube.com or youtu.be URL.";
    return false;
  }
}

async function submitJob(event) {
  event.preventDefault();
  elements.submitError.textContent = "";
  if (!validateUrl()) return;

  const form = new FormData(elements.form);
  const introValue = form.get("intro_offset");
  const payload = {
    url: String(form.get("url")).trim(),
    backend: form.get("backend"),
    voice: form.get("voice"),
    max_segments: Number(form.get("max_segments")),
    preserve_non_speech: form.has("preserve_non_speech"),
    reuse_translation: form.has("reuse_translation"),
    keep_original_audio: form.has("keep_original_audio"),
    original_volume: 0.1,
    intro_offset: introValue === "" ? null : Number(introValue),
  };

  elements.runButton.disabled = true;
  elements.runButton.classList.add("loading");
  elements.runButton.querySelector("span").textContent = "Creating run";

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail) ? result.detail[0]?.msg : result.detail;
      throw new Error(detail || "Could not create the run");
    }

    state.events = [];
    state.transcriptCache = {};
    state.activeLanguage = "english";
    state.activePhaseId = "download";
    state.phasePinned = false;
    elements.transcriptPanel.hidden = true;
    renderEvents();
    renderSnapshot(result);
    connectToJob(result.id);
    showToast(`Run ${result.id.toUpperCase()} added to the local queue`);
  } catch (error) {
    elements.submitError.textContent = error.message;
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.classList.remove("loading");
    elements.runButton.querySelector("span").textContent = "Start processing";
  }
}

function updateVoiceOptions(backend) {
  const voices = state.voices[backend] || FALLBACK_VOICES[backend];
  elements.voice.replaceChildren();
  voices.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = titleCase(voice);
    elements.voice.append(option);
  });
  renderVoiceProfile();
}

function renderVoiceProfile() {
  const backend = document.querySelector('input[name="backend"]:checked').value;
  const profile = state.voiceProfiles[backend]?.[elements.voice.value];
  if (!profile) {
    elements.voiceProfileMeta.textContent = titleCase(backend);
    elements.voiceProfileDescription.textContent = "Voice profile details are unavailable.";
    return;
  }
  elements.voiceProfileMeta.textContent = `${profile.gender} · ${profile.tone}`;
  elements.voiceProfileDescription.textContent = profile.description;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const health = await response.json();
    state.voices = health.voices || FALLBACK_VOICES;
    state.voiceProfiles = health.voice_profiles || FALLBACK_VOICE_PROFILES;
    let backend = document.querySelector('input[name="backend"]:checked').value;
    if (!health.services[backend]) {
      backend = health.services.gemini ? "gemini" : "edge";
      document.querySelector(`input[name="backend"][value="${backend}"]`).checked = true;
    }
    updateVoiceOptions(backend);
    elements.systemLabel.textContent = health.status === "ready" ? "Runtime ready" : "Check runtime";
    elements.systemPill.querySelector(".status-dot").classList.toggle("attention", health.status !== "ready");
    elements.serviceList.replaceChildren();
    Object.entries(health.services).forEach(([name, ready]) => {
      const row = document.createElement("div");
      row.className = "service-row";
      const serviceName = document.createElement("span");
      serviceName.textContent = titleCase(name);
      const status = document.createElement("span");
      status.textContent = ready ? "Ready" : "Not configured";
      status.classList.toggle("off", !ready);
      row.append(serviceName, status);
      elements.serviceList.append(row);
    });
  } catch (_) {
    elements.systemLabel.textContent = "Server unavailable";
    elements.systemPill.querySelector(".status-dot").classList.add("attention");
  }
}

async function loadHistory() {
  try {
    const response = await fetch("/api/jobs");
    if (!response.ok) return;
    const jobs = await response.json();
    elements.historyCount.textContent = jobs.length;
    elements.historyList.replaceChildren();
    if (!jobs.length) {
      const empty = document.createElement("p");
      empty.className = "history-empty";
      empty.textContent = "Completed runs will remain here during this server session.";
      elements.historyList.append(empty);
      return;
    }
    jobs.forEach((job) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `history-item ${job.status}`;
      button.dataset.jobId = job.id;
      const copy = document.createElement("span");
      copy.className = "history-copy";
      const title = document.createElement("strong");
      title.textContent = job.result?.title || `Run ${job.id.toUpperCase()}`;
      const backend = document.createElement("small");
      backend.textContent = `${job.request.backend} · ${job.request.voice}`;
      copy.append(title, backend);
      const status = document.createElement("small");
      status.textContent = job.status;
      button.append(copy, status);
      button.addEventListener("click", () => selectHistoryJob(job.id));
      elements.historyList.append(button);
    });
  } catch (_) {
    // History is useful but non-critical to processing.
  }
}

async function selectHistoryJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) throw new Error("Run is no longer available");
    state.events = [];
    state.transcriptCache = {};
    renderEvents();
    const snapshot = await response.json();
    renderSnapshot(snapshot);
    if (!["complete", "failed"].includes(snapshot.status)) connectToJob(jobId);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4400);
}

function updateElapsed() {
  if (!state.job?.started_at) {
    elements.elapsedTime.textContent = "00:00";
    return;
  }
  const end = state.job.finished_at || Date.now() / 1000;
  elements.elapsedTime.textContent = formatDuration(end - state.job.started_at);
  if (state.job.status === "running") renderPhases();
}

elements.form.addEventListener("submit", submitJob);
elements.sampleRun.addEventListener("click", loadSampleRun);
elements.url.addEventListener("input", () => { elements.urlError.textContent = ""; });
document.querySelectorAll('input[name="backend"]').forEach((radio) => {
  radio.addEventListener("change", () => updateVoiceOptions(radio.value));
});
elements.voice.addEventListener("change", renderVoiceProfile);
elements.clearLog.addEventListener("click", () => {
  state.events = [];
  renderEvents();
});
elements.followPhase.addEventListener("click", () => {
  state.phasePinned = false;
  const livePhase = state.job?.current_phase;
  if (livePhase) state.activePhaseId = livePhase;
  renderPhases();
});
elements.systemPill.addEventListener("click", () => {
  const open = elements.systemPopover.hidden;
  elements.systemPopover.hidden = !open;
  elements.systemPill.setAttribute("aria-expanded", String(open));
});
elements.closeSystem.addEventListener("click", () => {
  elements.systemPopover.hidden = true;
  elements.systemPill.setAttribute("aria-expanded", "false");
});
document.querySelectorAll("[data-language]").forEach((button) => {
  button.addEventListener("click", () => loadTranscript(button.dataset.language));
});

renderPhases();
renderEvents();
updateVoiceOptions("sarvam");
loadHealth();
loadHistory();
window.setInterval(updateElapsed, 1000);
