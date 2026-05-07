"""Optional LLM second pass for citation alignment.

By default, bib-checker is fully algorithmic — nothing here is invoked. When
the user passes ``--llm <provider>`` on the CLI, each citation marked for review
is sent to the chosen backend with a structured prompt and the LLM's verdict is
attached to the result. Three providers are supported:

    none       — no LLM (default)
    ollama     — local Ollama server (default URL http://localhost:11434, no key)
    anthropic  — Claude API (requires ANTHROPIC_API_KEY env var)
    openai     — OpenAI API (requires OPENAI_API_KEY env var)

The cloud providers send the citation context, the cited paper's title, and (if
available) its abstract. That is the user's content, opted into via the flag.
"""

from __future__ import annotations
import json
import os
import re
import requests


PROMPT = """You are reviewing whether a citation in a paper is appropriate.

Read the passage from the paper and the cited work's title and abstract.
Decide whether the cited work supports the specific claim made around the
citation marker. Common failure modes to flag:
- Cited paper is on a different topic.
- Cited paper's findings contradict the claim (polarity flip).
- Cited paper is tangentially related but does not support the specific claim.
- Cited paper is a foundational reference being applied too narrowly or too broadly.

Passage from paper:
\"\"\"
{context}
\"\"\"

Cited work:
Title: {title}
Abstract: {abstract}

Reply with a single JSON object and nothing else:
{{"verdict": "support" | "tangential" | "mismatch" | "unknown",
  "confidence": <float 0.0-1.0>,
  "reason": "<one short sentence>"}}"""


def _parse_json(raw: str) -> dict:
    """Try to extract a JSON object from the LLM response."""
    if not raw:
        return {"verdict": "unknown", "confidence": 0.0, "reason": "empty response"}
    raw = raw.strip()
    # Strip ```json fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"verdict": "unknown", "confidence": 0.0, "reason": f"unparseable: {raw[:200]}"}


class LLMClient:
    """Base interface — all subclasses implement .review()."""

    def review(self, context: str, title: str, abstract: str) -> dict:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def review(self, context: str, title: str, abstract: str) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": PROMPT.format(
                    context=context[:2000], title=title, abstract=(abstract or "(not available)")[:2000]
                )}
            ],
            "stream": False,
            "format": "json",
        }
        r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        msg = r.json().get("message", {}).get("content", "")
        return _parse_json(msg)


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY env var is required for --llm anthropic")

    def review(self, context: str, title: str, abstract: str) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": PROMPT.format(
                context=context[:2000], title=title, abstract=(abstract or "(not available)")[:2000]
            )}],
        }
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=120)
        r.raise_for_status()
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return _parse_json(text)


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY env var is required for --llm openai")

    def review(self, context: str, title: str, abstract: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(
                context=context[:2000], title=title, abstract=(abstract or "(not available)")[:2000]
            )}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=120)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]["content"]
        return _parse_json(msg)


def make_client(provider: str, model: str | None = None, base_url: str | None = None) -> LLMClient | None:
    """Return an LLMClient for the named provider, or None for 'none'."""
    p = (provider or "none").lower()
    if p == "none":
        return None
    if p == "ollama":
        return OllamaClient(model=model or "llama3.1", base_url=base_url)
    if p == "anthropic":
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001")
    if p == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", base_url=base_url)
    raise ValueError(f"Unknown LLM provider: {provider}")
