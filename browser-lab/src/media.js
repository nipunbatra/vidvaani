import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import coreURL from "@ffmpeg/core?url";
import wasmURL from "@ffmpeg/core/wasm?url";

export class MediaEngine {
  constructor() {
    this.ffmpeg = null;
    this.loaded = false;
    this.progressHandler = null;
  }

  async load(onProgress) {
    this.progressHandler = onProgress;
    if (this.loaded) return;
    this.ffmpeg = new FFmpeg();
    this.ffmpeg.on("progress", ({ progress }) => this.progressHandler?.(progress));
    await this.ffmpeg.load({ coreURL, wasmURL });
    this.loaded = true;
  }

  async extractAudio(file, onProgress) {
    await this.load(onProgress);
    this.progressHandler = onProgress;
    const extension = safeExtension(file.name, file.type);
    const inputName = `source.${extension}`;
    const audioName = "source-16k.wav";
    await this.safeDelete(inputName);
    await this.safeDelete(audioName);
    await this.ffmpeg.writeFile(inputName, await fetchFile(file));
    const code = await this.ffmpeg.exec([
      "-i", inputName,
      "-vn",
      "-ac", "1",
      "-ar", "16000",
      "-c:a", "pcm_s16le",
      audioName,
    ]);
    if (code !== 0) throw new Error("Could not extract audio from this media file.");
    const audioBytes = await this.ffmpeg.readFile(audioName);
    await this.safeDelete(audioName);
    return { inputName, audioBytes: Uint8Array.from(audioBytes) };
  }

  async assemble(inputName, file, audioSegments, duration, onProgress) {
    if (!this.loaded || !this.ffmpeg) throw new Error("Media engine is not ready.");
    this.progressHandler = onProgress;
    const segmentNames = [];
    for (let index = 0; index < audioSegments.length; index += 1) {
      const name = `dub-segment-${index}.wav`;
      segmentNames.push(name);
      await this.ffmpeg.writeFile(name, await fetchFile(audioSegments[index].blob));
    }

    const inputs = ["-i", inputName];
    for (const name of segmentNames) inputs.push("-i", name);

    const delayed = audioSegments.map((segment, index) => {
      const delay = Math.max(0, Math.round(segment.start * 1000));
      return `[${index + 1}:a]adelay=${delay}:all=1[a${index}]`;
    });
    const mixInputs = audioSegments.map((_, index) => `[a${index}]`).join("");
    delayed.push(
      `${mixInputs}amix=inputs=${audioSegments.length}:duration=longest:normalize=0,apad,atrim=0:${duration.toFixed(3)}[dub]`,
    );

    const isVideo = file.type.startsWith("video/");
    const isWebM = file.type === "video/webm" || inputName.endsWith(".webm");
    const outputName = isVideo ? (isWebM ? "vidvaani-dub.webm" : "vidvaani-dub.mp4") : "vidvaani-dub.wav";
    const args = [
      ...inputs,
      "-filter_complex", delayed.join(";"),
    ];
    if (isVideo) {
      args.push(
        "-map", "0:v:0",
        "-map", "[dub]",
        "-c:v", "copy",
        "-c:a", isWebM ? "libopus" : "aac",
        "-b:a", "160k",
        "-t", duration.toFixed(3),
      );
      if (!isWebM) args.push("-movflags", "+faststart");
    } else {
      args.push("-map", "[dub]", "-c:a", "pcm_s16le");
    }
    args.push(outputName);

    const code = await this.ffmpeg.exec(args);
    if (code !== 0) throw new Error("Local video assembly failed for this codec combination.");
    const outputBytes = Uint8Array.from(await this.ffmpeg.readFile(outputName));
    const mimeType = isVideo ? (isWebM ? "video/webm" : "video/mp4") : "audio/wav";

    for (const name of segmentNames) await this.safeDelete(name);
    await this.safeDelete(outputName);
    await this.safeDelete(inputName);
    return { blob: new Blob([outputBytes], { type: mimeType }), mimeType, outputName };
  }

  async safeDelete(path) {
    try {
      await this.ffmpeg?.deleteFile(path);
    } catch {
      // A missing in-memory file requires no action.
    }
  }

  reset() {
    this.ffmpeg?.terminate();
    this.ffmpeg = null;
    this.loaded = false;
    this.progressHandler = null;
  }
}

export async function probeMedia(file) {
  const element = document.createElement(file.type.startsWith("audio/") ? "audio" : "video");
  const url = URL.createObjectURL(file);
  element.preload = "metadata";
  element.src = url;
  try {
    await new Promise((resolve, reject) => {
      element.onloadedmetadata = resolve;
      element.onerror = () => reject(new Error("The browser could not read this media file."));
    });
    return {
      duration: Number(element.duration),
      width: "videoWidth" in element ? element.videoWidth : 0,
      height: "videoHeight" in element ? element.videoHeight : 0,
    };
  } finally {
    element.removeAttribute("src");
    URL.revokeObjectURL(url);
  }
}

export async function decodeWav(audioBytes) {
  const context = new AudioContext({ sampleRate: 16000 });
  try {
    const buffer = await context.decodeAudioData(audioBytes.buffer.slice(0));
    return Float32Array.from(buffer.getChannelData(0));
  } finally {
    await context.close();
  }
}

export function floatAudioToWav(audio, sampleRate) {
  const pcm = new Uint8Array(audio.length * 2);
  const view = new DataView(pcm.buffer);
  for (let index = 0; index < audio.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, audio[index]));
    view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return pcm16ToWav(pcm, sampleRate);
}

export function pcm16ToWav(pcmBytes, sampleRate = 24000) {
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + pcmBytes.byteLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, pcmBytes.byteLength, true);
  return new Blob([header, pcmBytes], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function safeExtension(name, mimeType) {
  const extension = name.split(".").pop()?.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (extension && extension.length <= 5) return extension;
  if (mimeType === "video/webm") return "webm";
  if (mimeType.startsWith("audio/")) return "wav";
  return "mp4";
}
