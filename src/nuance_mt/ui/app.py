"""Gradio demo with clickable cultural-nuance info boxes."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from nuance_mt.pipeline import NuancePipeline, highlight_html


@lru_cache(maxsize=1)
def get_pipeline() -> NuancePipeline:
    return NuancePipeline()


CSS = """
mark.nuance {
  background: linear-gradient(120deg, #f6e7c1 0%, #e8d4a8 100%);
  padding: 0 0.15em;
  border-radius: 2px;
  cursor: pointer;
  border-bottom: 2px solid #8a6a2f;
}
.nuance-panel {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif;
  line-height: 1.45;
}
"""


def analyze(text: str, target_label: str):
    if not text or not text.strip():
        return "Enter English source text.", "", "No spans yet."
    target = "de" if target_label.startswith("German") else "ja"
    pipe = get_pipeline()
    result = pipe.run(text.strip(), target)
    html = (
        "<div class='nuance-panel'><p><b>Source (click highlights)</b></p>"
        f"<p>{highlight_html(result.source, result.hits)}</p></div>"
    )
    if not result.hits:
        detail = (
            "_No cultural-nuance spans detected above threshold._ "
            "Try seed examples like *homeland*, *cozy*, *dependence*, or *reading the room*."
        )
    else:
        blocks = []
        for i, h in enumerate(result.hits):
            blocks.append(
                f"### [{i}] `{h.text}` — **{h.nuance_type}** (score {h.score:.2f})\n"
                f"- **English frame:** {h.en_frame or '—'}\n"
                f"- **Target frame:** {h.target_frame or '—'}\n"
                f"- **Suggested target span:** {h.tgt_span or '—'}\n\n"
                f"{h.explanation or '_No explanation in store for this span._'}\n\n"
                f"*Evidence:* {h.evidence or '—'}"
            )
        detail = "\n\n---\n\n".join(blocks)
    return html, result.translation, detail


def main():
    examples = [
        [
            "After years abroad, he still longed for his homeland in a way that 'home country' never quite captured.",
            "German (de)",
        ],
        [
            "The evening had a cozy warmth, but it was more than comfort — a shared Gemütlichkeit of belonging.",
            "German (de)",
        ],
        [
            "Her dependence on her mentor was not weakness; it was a culturally legible form of intimacy.",
            "Japanese (ja)",
        ],
        [
            "He was good at reading the room before speaking, prioritizing group harmony over blunt honesty.",
            "Japanese (ja)",
        ],
    ]

    with gr.Blocks(title="Cultural Nuance Translation", css=CSS) as demo:
        gr.Markdown(
            """
# Cultural Nuance Translation
**EN → DE / EN → JA** — translation plus research-backed cultural asymmetry notes.

Highlights mark English spans where denotation matches but cultural framing diverges.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="English source", lines=5)
                target = gr.Radio(
                    ["German (de)", "Japanese (ja)"],
                    value="German (de)",
                    label="Target language",
                )
                btn = gr.Button("Translate + detect nuances", variant="primary")
            with gr.Column(scale=1):
                src_html = gr.HTML(label="Highlighted source")
                translation = gr.Textbox(label="Translation (NLLB)", lines=5)
        detail = gr.Markdown(label="Nuance info")
        btn.click(analyze, inputs=[text, target], outputs=[src_html, translation, detail])
        gr.Examples(examples=examples, inputs=[text, target])

    demo.launch()


if __name__ == "__main__":
    main()
