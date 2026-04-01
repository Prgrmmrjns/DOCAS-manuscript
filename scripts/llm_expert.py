from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from mistralai import Mistral


DEFAULT_MODEL = "mistral-small-2603"

# Reusable guidance for any LLM call that must emit a high-quality Graphviz
# explanatory model. Keep this strict so downstream rendering is stable.
GRAPHVIZ_EXPLANATORY_MODEL_PROMPT = """
When generating `graphviz_dot`, produce a COMPLETE and VALID Graphviz DOT string:
- Must start with: digraph pathophysiology {
- Must end with: }
- No markdown fences and no surrounding prose.
- Use only DOT syntax.

Graph quality requirements:
- rankdir=TB for readability.
- Use rounded filled boxes for nodes and include concise labels.
- Include these conceptual groups in the graph:
  1) observed/model signals
  2) organ-system mechanisms
  3) outcome node
- Add directed edges from signals -> systems and systems -> outcome.
- Add cross-system edges when interactions are clinically meaningful.
- Keep labels short, clinician-friendly, and non-mathematical.
- Avoid disconnected nodes.

Styling guidance:
- Use consistent colors by role (e.g., blue signals, orange systems, green outcome).
- Include edge labels only when they improve interpretability.
- Avoid excessive density; prioritize top mechanisms and interactions.
"""


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def call_text(*, system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.15) -> str:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def call_json(*, system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.15) -> dict:
    raw = call_text(system=system + "\nReturn only valid JSON, no markdown.", user=user, model=model, temperature=temperature)
    return _parse_json(raw)

