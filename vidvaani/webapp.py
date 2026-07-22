"""Professional web control room for the VidVaani processing pipeline."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator


PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
ASSET_DIR = WEB_DIR / "assets"
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR.parent / "output" / "web"
DEMO_ASSET_DIR = PACKAGE_DIR.parent / "demo_videos"

PHASES = (
    {"id": "download", "label": "Source media", "engine": "yt-dlp", "weight": 0.12},
    {"id": "analyze", "label": "Intro analysis", "engine": "ffmpeg", "weight": 0.05},
    {"id": "transcribe", "label": "Speech to text", "engine": "MLX Whisper", "weight": 0.24},
    {"id": "translate", "label": "Hindi translation", "engine": "Gemini Flash", "weight": 0.17},
    {"id": "synthesize", "label": "Voice synthesis", "engine": "TTS", "weight": 0.27},
    {"id": "assemble", "label": "Timing & mix", "engine": "ffmpeg", "weight": 0.13},
    {"id": "deliver", "label": "Deliverables", "engine": "MP4 · SRT · JSON", "weight": 0.02},
)

VOICE_PROFILES = {
    "sarvam": {
        "abhilash": {"gender": "Male", "tone": "Steady lecture tone", "description": "Native Indian prosody; measured and dependable for technical explanations."},
        "anushka": {"gender": "Female", "tone": "Warm and clear", "description": "Native Indian prosody; an expressive narration voice with crisp diction."},
        "arya": {"gender": "Female", "tone": "Balanced", "description": "Native Indian prosody; a neutral alternative for long-form lessons."},
        "hitesh": {"gender": "Male", "tone": "Direct", "description": "Native Indian prosody; a distinct male timbre for lecture material."},
        "karun": {"gender": "Male", "tone": "Conversational", "description": "Native Indian prosody; a softer alternative for explanatory teaching."},
        "manisha": {"gender": "Female", "tone": "Composed", "description": "Native Indian prosody; an even female voice for sustained narration."},
        "vidya": {"gender": "Female", "tone": "Clear lecture delivery", "description": "Native Indian prosody; a clean, classroom-friendly female voice."},
    },
    "gemini": {
        "Aoede": {"gender": "Female", "tone": "Breezy", "description": "Google's breezy voice profile; light delivery for accessible narration."},
        "Charon": {"gender": "Male", "tone": "Informative", "description": "Google's informative profile; natural delivery with a slight Western accent in Hindi."},
        "Fenrir": {"gender": "Male", "tone": "Excitable", "description": "Google's energetic profile; more animated than a conventional lecture voice."},
        "Iapetus": {"gender": "Male", "tone": "Clear", "description": "Google's clear profile; precise diction and an even pace."},
        "Kore": {"gender": "Female", "tone": "Firm", "description": "Google's firm profile; confident, structured delivery."},
        "Orus": {"gender": "Male", "tone": "Firm", "description": "Google's firm profile; confident and well suited to lecture narration."},
        "Puck": {"gender": "Male", "tone": "Upbeat", "description": "Google's upbeat profile; lively delivery for shorter material."},
        "Sadaltager": {"gender": "Male", "tone": "Knowledgeable", "description": "Google's knowledgeable profile; warm, authoritative delivery."},
    },
    "edge": {
        "male": {"gender": "Male", "tone": "Madhur", "description": "Microsoft hi-IN-MadhurNeural; free Hindi fallback with a male voice."},
        "female": {"gender": "Female", "tone": "Swara", "description": "Microsoft hi-IN-SwaraNeural; free Hindi fallback with a female voice."},
    },
}
VOICE_OPTIONS = {backend: tuple(profiles) for backend, profiles in VOICE_PROFILES.items()}

PIPELINE_IMPORTS = ("rich", "mlx_whisper", "google.genai", "edge_tts", "pydub")


def _missing_pipeline_modules() -> list[str]:
    """Return modules missing from the interpreter serving this web app."""
    return [name for name in PIPELINE_IMPORTS if importlib.util.find_spec(name) is None]


class JobRequest(BaseModel):
    """Validated settings for one dubbing run."""

    url: str = Field(min_length=12, max_length=2048)
    backend: Literal["gemini", "sarvam", "edge"] = "sarvam"
    voice: str = "abhilash"
    max_segments: int = Field(default=5, ge=-1, le=250)
    keep_original_audio: bool = False
    original_volume: float = Field(default=0.1, ge=0.0, le=0.5)
    preserve_non_speech: bool = True
    reuse_translation: bool = True
    intro_offset: float | None = Field(default=None, ge=0.0, le=600.0)

    @model_validator(mode="after")
    def validate_source_and_voice(self) -> "JobRequest":
        parsed = urlparse(self.url)
        host = (parsed.hostname or "").lower()
        allowed_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
        if parsed.scheme not in {"http", "https"} or host not in allowed_hosts:
            raise ValueError("Enter a valid youtube.com or youtu.be URL")
        if self.voice not in VOICE_OPTIONS[self.backend]:
            raise ValueError(f"Voice {self.voice!r} is not available for {self.backend}")
        return self


class ProcessingJob:
    """Thread-safe in-memory state for a pipeline run."""

    def __init__(self, request: JobRequest) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.current_phase: str | None = None
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.costs: dict[str, Any] = {"total_cost_usd": 0.0, "total_cost_inr": 0.0}
        self.artifact_paths: dict[str, Path] = {}
        self.phases = {
            phase["id"]: {
                "id": phase["id"],
                "label": phase["label"],
                "engine": phase["engine"],
                "weight": phase["weight"],
                "status": "pending",
                "progress": 0.0,
                "message": "Waiting",
                "started_at": None,
                "finished_at": None,
                "details": {},
            }
            for phase in PHASES
        }
        self._events: list[dict[str, Any]] = []
        self._event_sequence = 0
        self._lock = threading.RLock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            phase_id = event.get("phase")
            status = event.get("status", "running")
            now = float(event.get("timestamp", time.time()))

            if phase_id in self.phases:
                phase = self.phases[phase_id]
                if phase["started_at"] is None and status in {"running", "complete", "cached"}:
                    phase["started_at"] = now
                phase["status"] = status
                phase["progress"] = float(event.get("progress", phase["progress"]))
                phase["message"] = event.get("message", phase["message"])
                phase["details"].update(event.get("details") or {})
                if status in {"complete", "cached", "failed"}:
                    phase["finished_at"] = now
                    if status in {"complete", "cached"}:
                        phase["progress"] = 1.0
                if status == "running":
                    self.current_phase = phase_id

            reported_costs = (event.get("details") or {}).get("costs")
            if reported_costs:
                self.costs = copy.deepcopy(reported_costs)

            if phase_id == "pipeline" and status == "running":
                self.status = "running"
                self.started_at = self.started_at or now

            self._event_sequence += 1
            enriched = copy.deepcopy(event)
            enriched["id"] = self._event_sequence
            enriched["job_status"] = self.status
            self._events.append(enriched)
            # A long download can produce many updates. Retain a useful bounded history.
            if len(self._events) > 1200:
                self._events = self._events[-900:]

    def finish(self, result: dict[str, Any], artifacts: dict[str, Path]) -> None:
        with self._lock:
            self.result = result
            self.costs = copy.deepcopy(result.get("costs") or self.costs)
            self.artifact_paths = artifacts
            self.status = "complete"
            self.current_phase = "deliver"
            self.finished_at = time.time()

    def fail(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self.error = message
            self.status = "failed"
            self.finished_at = now
            if self.current_phase and self.current_phase in self.phases:
                phase = self.phases[self.current_phase]
                phase["status"] = "failed"
                phase["message"] = message
                phase["finished_at"] = now
            self._event_sequence += 1
            self._events.append({
                "id": self._event_sequence,
                "phase": self.current_phase or "pipeline",
                "status": "failed",
                "progress": 0.0,
                "message": message,
                "timestamp": now,
                "details": {},
                "job_status": "failed",
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            overall = sum(
                phase["weight"] * phase["progress"]
                for phase in self.phases.values()
            )
            artifacts = [
                {
                    "key": key,
                    "name": path.name,
                    "size": path.stat().st_size if path.exists() else 0,
                    "url": f"/api/jobs/{self.id}/artifacts/{key}",
                }
                for key, path in self.artifact_paths.items()
            ]
            return copy.deepcopy({
                "id": self.id,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "current_phase": self.current_phase,
                "progress": overall,
                "error": self.error,
                "request": self.request.model_dump(),
                "phases": list(self.phases.values()),
                "result": self.result,
                "costs": self.costs,
                "artifacts": artifacts,
            })

    def events_after(self, sequence: int) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy([event for event in self._events if event["id"] > sequence])


class JobRegistry:
    def __init__(self, max_jobs: int = 20) -> None:
        self.max_jobs = max_jobs
        self._jobs: dict[str, ProcessingJob] = {}
        self._lock = threading.RLock()

    def create(self, request: JobRequest) -> ProcessingJob:
        job = ProcessingJob(request)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > self.max_jobs:
                terminal = [
                    item for item in self._jobs.values()
                    if item.status in {"complete", "failed"} and item.id != job.id
                ]
                if terminal:
                    oldest = min(terminal, key=lambda item: item.created_at)
                    self._jobs.pop(oldest.id, None)
        return job

    def get(self, job_id: str) -> ProcessingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Processing job not found")
        return job

    def recent(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return [job.snapshot() for job in jobs]


registry = JobRegistry()
# MLX transcription and video assembly are resource-heavy; queue runs one at a time.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vidvaani-web")


def _artifact_map(result: Any) -> dict[str, Path]:
    candidates = {
        "video": result.output_video,
        "subtitles": result.subtitle_path,
        "english": result.transcript_path,
        "hindi": result.translation_path,
    }
    return {
        key: Path(path).resolve()
        for key, path in candidates.items()
        if path is not None and Path(path).exists()
    }


def _run_processing_job(job: ProcessingJob) -> None:
    output_root = Path(os.environ.get("VIDVAANI_WEB_OUTPUT", DEFAULT_OUTPUT_ROOT)).expanduser()
    output_dir = output_root / job.id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import lazily so the control room and health check remain lightweight.
        from .pipeline import run_pipeline

        settings = job.request
        result = run_pipeline(
            url=settings.url,
            output_dir=output_dir,
            voice=settings.voice,
            tts_backend=settings.backend,
            keep_original_audio=settings.keep_original_audio,
            original_volume=settings.original_volume,
            max_segments=settings.max_segments,
            reuse_translation=settings.reuse_translation,
            preserve_non_speech=settings.preserve_non_speech,
            intro_offset=settings.intro_offset,
            event_callback=job.publish,
        )

        deliver_started = time.time()
        job.publish({
            "phase": "deliver",
            "status": "running",
            "progress": 0.35,
            "message": "Indexing video, subtitles, and transcripts",
            "timestamp": time.time(),
            "details": {},
        })
        artifacts = _artifact_map(result)
        summary = {
            "title": result.title,
            "duration": result.duration,
            "segments": result.segments_count,
            "timings": result.timings or {},
            "costs": result.costs or {},
        }
        job.publish({
            "phase": "deliver",
            "status": "complete",
            "progress": 1.0,
            "message": f"Ready — {len(artifacts)} files available",
            "timestamp": time.time(),
            "details": {
                "artifact_count": len(artifacts),
                "artifact_types": list(artifacts),
                "duration_seconds": time.time() - deliver_started,
            },
        })
        job.finish(summary, artifacts)
    except Exception as exc:
        job.fail(str(exc) or exc.__class__.__name__)


app = FastAPI(
    title="VidVaani Control Room",
    description="Observable browser interface for English-to-Hindi video dubbing",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")
app.mount("/demo-assets", StaticFiles(directory=DEMO_ASSET_DIR), name="demo-assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    sarvam_ready = bool(os.environ.get("SARVAM_API_KEY"))
    missing_modules = _missing_pipeline_modules()
    ffmpeg_ready = bool(shutil.which("ffmpeg"))
    ffprobe_ready = bool(shutil.which("ffprobe"))
    return {
        "status": "ready" if ffmpeg_ready and ffprobe_ready and not missing_modules else "attention",
        "services": {
            "pipeline": not missing_modules,
            "ffmpeg": ffmpeg_ready,
            "ffprobe": ffprobe_ready,
            "gemini": gemini_ready,
            "sarvam": sarvam_ready,
            "edge": True,
        },
        "voices": VOICE_OPTIONS,
        "voice_profiles": VOICE_PROFILES,
        "missing_modules": missing_modules,
        "queue_policy": "one active run; additional runs wait in order",
    }


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest) -> dict[str, Any]:
    missing_modules = _missing_pipeline_modules()
    if missing_modules:
        missing = ", ".join(missing_modules)
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline dependencies missing: {missing}. Start the app with 'uv run --frozen vidvaani-web'.",
        )
    job = registry.create(request)
    executor.submit(_run_processing_job, job)
    return job.snapshot()


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return registry.recent()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return registry.get(job_id).snapshot()


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(
    job_id: str,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    job = registry.get(job_id)

    async def event_stream():
        sequence = after
        heartbeat_at = time.monotonic()
        while True:
            events = job.events_after(sequence)
            for event in events:
                sequence = event["id"]
                yield f"id: {sequence}\nevent: progress\ndata: {json.dumps(event)}\n\n"

            if job.status in {"complete", "failed"} and not job.events_after(sequence):
                snapshot = job.snapshot()
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
                break

            if time.monotonic() - heartbeat_at > 12:
                yield ": keep-alive\n\n"
                heartbeat_at = time.monotonic()
            await asyncio.sleep(0.35)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str) -> FileResponse:
    job = registry.get(job_id)
    path = job.artifact_paths.get(artifact_key)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not available")
    media_types = {
        "video": "video/mp4",
        "subtitles": "application/x-subrip",
        "english": "application/json",
        "hindi": "application/json",
    }
    return FileResponse(path, media_type=media_types.get(artifact_key), filename=path.name)


@app.get("/api/jobs/{job_id}/transcript/{language}")
def get_transcript(job_id: str, language: Literal["english", "hindi"]) -> JSONResponse:
    job = registry.get(job_id)
    path = job.artifact_paths.get(language)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not available")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    import uvicorn

    host = os.environ.get("VIDVAANI_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("VIDVAANI_WEB_PORT", "7860"))
    uvicorn.run("vidvaani.webapp:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
