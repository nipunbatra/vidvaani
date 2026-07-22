"""Focused tests for the observable web control room."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from vidvaani.webapp import JobRequest, ProcessingJob, app, registry
from vidvaani.translator import translate_segments
from vidvaani.video import AssemblyResult, create_hindi_video


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_index_and_health_are_available(self) -> None:
        index = self.client.get("/")
        health = self.client.get("/api/health")

        self.assertEqual(index.status_code, 200)
        self.assertIn("VidVaani", index.text)
        self.assertEqual(
            self.client.get("/demo-assets/mini_demo/sarvam_abhilash.mp4", headers={"range": "bytes=0-31"}).status_code,
            206,
        )
        self.assertEqual(health.status_code, 200)
        self.assertIn("services", health.json())
        self.assertIn("voices", health.json())
        self.assertTrue(health.json()["services"]["pipeline"])
        self.assertEqual(health.json()["missing_modules"], [])

    def test_invalid_source_is_rejected_before_queueing(self) -> None:
        response = self.client.post("/api/jobs", json={
            "url": "https://example.com/video",
            "backend": "sarvam",
            "voice": "abhilash",
        })

        self.assertEqual(response.status_code, 422)

    def test_phase_events_produce_a_weighted_snapshot(self) -> None:
        job = ProcessingJob(JobRequest(
            url="https://www.youtube.com/watch?v=4TC5s_xNKSs",
            backend="edge",
            voice="male",
        ))
        job.publish({
            "phase": "pipeline",
            "status": "running",
            "progress": 0.0,
            "message": "Pipeline initialized",
            "timestamp": 100.0,
            "details": {},
        })
        job.publish({
            "phase": "download",
            "status": "complete",
            "progress": 1.0,
            "message": "Downloaded",
            "timestamp": 102.0,
            "details": {
                "duration": 58.2,
                "duration_seconds": 1.7,
                "width": 1920,
                "height": 1080,
                "costs": {"total_cost_usd": 0.0142, "total_cost_inr": 1.35},
            },
        })

        snapshot = job.snapshot()
        download = next(phase for phase in snapshot["phases"] if phase["id"] == "download")
        self.assertEqual(snapshot["status"], "running")
        self.assertAlmostEqual(snapshot["progress"], 0.12)
        self.assertEqual(download["status"], "complete")
        self.assertEqual(download["details"]["duration"], 58.2)
        self.assertEqual(download["details"]["duration_seconds"], 1.7)
        self.assertEqual(download["details"]["width"], 1920)
        self.assertEqual(snapshot["costs"]["total_cost_inr"], 1.35)

    @patch("vidvaani.translator.get_client", return_value=object())
    @patch("vidvaani.translator._translate_batch")
    def test_translation_reports_aligned_batch_previews(self, translate_batch, _client) -> None:
        translate_batch.return_value = [
            {
                "start": 0.0,
                "end": 3.2,
                "text": "We integrate over this region.",
                "translated": "हम इस क्षेत्र पर इंटीग्रेट करते हैं।",
            },
        ]
        updates = []

        translated = translate_segments(
            [SimpleNamespace(start=0.0, end=3.2, text="We integrate over this region.")],
            progress_callback=lambda batch, complete, total: updates.append((batch, complete, total)),
        )

        self.assertEqual(translated[0].translated, "हम इस क्षेत्र पर इंटीग्रेट करते हैं।")
        self.assertEqual(updates[0][0][0].original, "We integrate over this region.")
        self.assertEqual(updates[0][1:], (1, 1))

    @patch("vidvaani.video.replace_audio")
    @patch("vidvaani.video.create_mixed_audio")
    @patch("vidvaani.video.get_duration", return_value=684.5)
    def test_demo_assembly_stops_at_last_processed_segment(
        self, _duration, mixed_audio, replace_audio
    ) -> None:
        replace_audio.return_value = AssemblyResult(Path("dub.mp4"), 58.5, 1)

        create_hindi_video(
            Path("source.mp4"),
            [(Path("speech.mp3"), 46.1, 58.5)],
            Path("dub.mp4"),
            preserve_non_speech=True,
            output_duration=58.5,
        )

        self.assertEqual(mixed_audio.call_args.args[3], 58.5)

    def test_missing_pipeline_dependency_is_rejected_before_worker_start(self) -> None:
        with patch("vidvaani.webapp._missing_pipeline_modules", return_value=["rich"]):
            response = self.client.post("/api/jobs", json={
                "url": "https://www.youtube.com/watch?v=UuoVhUqWAFc",
                "backend": "edge",
                "voice": "male",
            })

        self.assertEqual(response.status_code, 503)
        self.assertIn("uv run --frozen vidvaani-web", response.json()["detail"])

    def test_real_pipeline_imports_in_serving_environment(self) -> None:
        from vidvaani.pipeline import run_pipeline

        self.assertTrue(callable(run_pipeline))
        self.assertIn("event_callback", run_pipeline.__annotations__)

    def test_terminal_job_stream_includes_event_and_snapshot(self) -> None:
        job = registry.create(JobRequest(
            url="https://youtu.be/4TC5s_xNKSs",
            backend="edge",
            voice="female",
        ))
        job.publish({
            "phase": "pipeline",
            "status": "running",
            "progress": 0.0,
            "message": "Pipeline initialized",
            "timestamp": 100.0,
            "details": {},
        })
        job.finish({"title": "Verified fixture"}, {})

        response = self.client.get(f"/api/jobs/{job.id}/events")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: progress", response.text)
        self.assertIn("event: snapshot", response.text)
        self.assertIn("Verified fixture", response.text)


if __name__ == "__main__":
    unittest.main()
