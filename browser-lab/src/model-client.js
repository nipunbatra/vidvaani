export class ModelClient {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.sequence = 0;
    this.pending = new Map();
    this.createWorker();
  }

  createWorker() {
    this.worker = new Worker(new URL("./model-worker.js", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", (event) => {
      const message = event.data;
      this.onEvent?.(message);
      const pending = this.pending.get(message.requestId);
      if (!pending) return;
      if (message.status === "complete") {
        this.pending.delete(message.requestId);
        pending.resolve(message.result);
      } else if (message.status === "error") {
        this.pending.delete(message.requestId);
        pending.reject(new Error(message.message));
      }
    });
    this.worker.addEventListener("error", () => {
      const error = new Error("The local model worker stopped unexpectedly.");
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    });
  }

  call(type, payload, transfer = []) {
    const requestId = `model-${++this.sequence}`;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage({ requestId, type, payload }, transfer);
    });
  }

  transcribe(audio, modelId, useWebGPU) {
    const transferableAudio = Float32Array.from(audio);
    return this.call(
      "transcribe",
      { audio: transferableAudio, modelId, useWebGPU },
      [transferableAudio.buffer],
    );
  }

  translate(segments, modelId) {
    return this.call("translate", { segments, modelId });
  }

  synthesize(segments, dtype) {
    return this.call("synthesize", { segments, dtype });
  }

  reset() {
    this.worker.terminate();
    const error = new DOMException("Run cancelled", "AbortError");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.createWorker();
  }
}
