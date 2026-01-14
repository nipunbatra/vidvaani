"""Gradio UI for YouTube Hindi Dubbing."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import gradio as gr

from .pipeline import run_pipeline
from .costs import CostTracker
from .tts import GEMINI_VOICES


def process_video(
    url: str,
    voice: str,
    max_segments: int,
    keep_original: bool,
    original_volume: float,
    progress=gr.Progress()
) -> tuple[str, str, str, str, str]:
    """Process video and return results.

    Returns:
        Tuple of (video_path, english_transcript, hindi_transcript, cost_info, status)
    """
    if not url:
        return None, "", "", "", "Please enter a YouTube URL"

    # Create temp output directory
    output_dir = Path(tempfile.mkdtemp(prefix="yt_hindi_"))

    progress(0.1, desc="Starting pipeline...")

    try:
        # Run pipeline
        result = run_pipeline(
            url=url,
            output_dir=output_dir,
            voice=voice,
            tts_backend="gemini",
            keep_original_audio=keep_original,
            original_volume=original_volume,
            max_segments=max_segments if max_segments > 0 else -1,
        )

        progress(0.9, desc="Finalizing...")

        # Read transcripts
        en_transcript = ""
        hi_transcript = ""

        if result.transcript_path and result.transcript_path.exists():
            with open(result.transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                en_transcript = "\n\n".join([
                    f"[{s['start']:.1f}s - {s['end']:.1f}s]\n{s['text']}"
                    for s in data.get("segments", [])
                ])

        if result.translation_path and result.translation_path.exists():
            with open(result.translation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                hi_transcript = "\n\n".join([
                    f"[{s['start']:.1f}s - {s['end']:.1f}s]\n{s['original']}\n→ {s['text']}"
                    for s in data.get("segments", [])
                ])

        # Cost info
        costs = result.costs or {}
        cost_info = f"""
**Translation:** {costs.get('translation', {}).get('input_tokens', 0):,} in / {costs.get('translation', {}).get('output_tokens', 0):,} out tokens = ${costs.get('translation', {}).get('cost_usd', 0):.4f}

**TTS:** {costs.get('tts', {}).get('characters', 0):,} chars ({costs.get('tts', {}).get('calls', 0)} calls) = ${costs.get('tts', {}).get('cost_usd', 0):.4f}

**Total:** ${costs.get('total_cost_usd', 0):.4f}
"""

        status = f"Done! Processed {result.segments_count} segments. Duration: {result.duration:.1f}s"

        progress(1.0, desc="Complete!")

        return str(result.output_video), en_transcript, hi_transcript, cost_info, status

    except Exception as e:
        return None, "", "", "", f"Error: {str(e)}"


def create_ui():
    """Create and return the Gradio interface."""

    with gr.Blocks(title="YouTube Hindi Dubbing", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # YouTube to Hindi Dubbing

        Convert YouTube videos to Hindi using:
        - **Whisper** (mlx) for transcription
        - **Gemini** for translation
        - **Gemini TTS** for voice synthesis
        """)

        with gr.Row():
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )

                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=GEMINI_VOICES,
                        value="Charon",
                        label="Voice"
                    )
                    segments_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=5,
                        step=1,
                        label="Max Segments (0 = all)"
                    )

                with gr.Row():
                    keep_original = gr.Checkbox(
                        label="Keep original audio (background)",
                        value=False
                    )
                    original_volume = gr.Slider(
                        minimum=0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        label="Original volume",
                        visible=False
                    )

                # Show/hide volume slider
                keep_original.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[keep_original],
                    outputs=[original_volume]
                )

                process_btn = gr.Button("Process Video", variant="primary", size="lg")
                status_text = gr.Markdown("")

            with gr.Column(scale=1):
                cost_info = gr.Markdown("**Costs will appear here**")

        with gr.Row():
            with gr.Column():
                output_video = gr.Video(label="Hindi Dubbed Video")

        with gr.Row():
            with gr.Column():
                en_transcript = gr.Textbox(
                    label="English Transcript",
                    lines=10,
                    max_lines=20
                )
            with gr.Column():
                hi_transcript = gr.Textbox(
                    label="Hindi Translation",
                    lines=10,
                    max_lines=20
                )

        # Process button click
        process_btn.click(
            fn=process_video,
            inputs=[url_input, voice_dropdown, segments_slider, keep_original, original_volume],
            outputs=[output_video, en_transcript, hi_transcript, cost_info, status_text]
        )

        # Examples
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=UuoVhUqWAFc", "Charon", 5],
            ],
            inputs=[url_input, voice_dropdown, segments_slider],
            label="Example"
        )

    return app


def main():
    """Launch the Gradio UI."""
    app = create_ui()
    app.launch(share=False)


if __name__ == "__main__":
    main()
