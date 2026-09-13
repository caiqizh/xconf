"""LLM clients for the model under test (black-box: text in, text out).

Three routes, picked by environment: Gemini via the Developer API
(GEMINI_API_KEY) or Vertex AI (GOOGLE_CLOUD_PROJECT); Claude via the
Anthropic API (ANTHROPIC_API_KEY) or Vertex; any open-weights model via an
OpenAI-compatible endpoint (VLLM_BASE_URL, e.g. vLLM). We only ever read
the text output -- no logits, no hidden states -- so the method stays
model-agnostic and black-box.

Robustness (elicitation is a long, API-bound job): retries distinguish
transient failures (429 rate-limit, 5xx, timeouts) -- exponential backoff
with jitter -- from fatal ones (e.g. 400), which fail fast. An
empty/blocked response after retries returns "" so the caller records an
unparseable (=> incorrect) item and moves on, rather than crashing the run
or retrying forever; interrupted runs resume cleanly (already-written ids
are skipped).
"""

from __future__ import annotations

import time
import random
import threading
from typing import Optional

from .config import Config


class LLMError(RuntimeError):
    pass


_RETRYABLE_CODES = {408, 409, 425, 429, 499, 500, 502, 503, 504}  # 499 CANCELLED: transient server cancel
_RETRYABLE_HINTS = (
    "429", "rate limit", "rate-limit", "resource exhausted", "resource_exhausted",
    "quota", "exceeded", "too many requests", "499", "cancelled", "500", "502", "503", "504",
    "unavailable", "deadline", "timeout", "timed out", "internal error", "overloaded",
    "connection", "reset by peer", "temporarily",
    "ssl", "eof occurred", "unexpected_eof", "broken pipe", "aborted",  # transient transport failures
    "bad file descriptor",  # Errno 9 under very high thread counts
)


def _is_retryable(e: Exception) -> bool:
    for attr in ("code", "status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            # a real status code is DECISIVE — never fall through to text hints (a 400 whose message
            # happens to contain a number like "55002" must not substring-match the "500" hint)
            return v in _RETRYABLE_CODES
    blob = (type(e).__name__ + " " + str(e)).lower()
    return any(h in blob for h in _RETRYABLE_HINTS)


class _RateLimiter:
    """Optional global min-interval throttle shared across worker threads."""

    def __init__(self, min_interval_s: float):
        self.min_interval = min_interval_s
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep = self._next - now
            if sleep > 0:
                time.sleep(sleep)
                now = time.monotonic()
            self._next = max(now, self._next) + self.min_interval


def _with_retry(fn, max_retries: int, limiter: _RateLimiter, label: str):
    last = None
    for attempt in range(max_retries):
        limiter.wait()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_retryable(e):
                raise LLMError(f"{label}: fatal non-retryable error: {e}") from e
            # exponential backoff with full jitter, capped
            backoff = min(60.0, 2.0 * (2 ** attempt)) * (0.5 + random.random())
            time.sleep(backoff)
    raise LLMError(f"{label}: exhausted {max_retries} retries; last error: {last}")


class GeminiClient:
    def __init__(self, cfg: Config):
        import os
        from google import genai
        from google.genai.types import HttpOptions

        self._cfg = cfg
        self._limiter = _RateLimiter(cfg.model.min_interval_s)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            # Gemini Developer API (aistudio.google.com): just an API key, no GCP project.
            self.client = genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
        else:
            # Vertex AI (the paper's runs): GOOGLE_CLOUD_PROJECT + application-default credentials.
            self.client = genai.Client(
                vertexai=True,
                project=cfg.vertex.project,
                location=cfg.vertex.location,
                http_options=HttpOptions(api_version="v1"),
            )

    def _gen_config(self, system: Optional[str], temperature: float, force_no_think: bool = False):
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        kwargs = dict(
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=self._cfg.model.max_output_tokens,
        )
        tb = self._cfg.model.thinking_budget
        import os as _os
        if _os.environ.get("XCONF_THINKING", "off") == "on" and not force_no_think:
            # thinking-ON ablation: BOUNDED budget so thinking + visible answer coexist; matches
            # practical deployment configs.
            kwargs["thinking_config"] = ThinkingConfig(thinking_budget=int(_os.environ.get("XCONF_THINK_BUDGET", "16384")))
        elif tb is not None and self._cfg.model.name.lower().startswith(("gemini-2.5", "gemini-3")):
            kwargs["thinking_config"] = ThinkingConfig(thinking_budget=tb)
        return GenerateContentConfig(**kwargs)

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
                 images: Optional[list] = None, prefix: Optional[str] = None,
                 force_no_think: bool = False) -> str:
        """images: optional list of (bytes, mime_type) tuples -> multimodal contents (Vertex Gemini).
        prefix: shared-prefix hint for prompt caching; Gemini caches implicitly, so just prepend.
        force_no_think: per-call opt-out from the XCONF_THINKING=on ablation."""
        temp = self._cfg.model.temperature if temperature is None else temperature
        if prefix:
            prompt = prefix + "\n" + prompt
        if images:
            from google.genai.types import Part
            contents = [Part.from_bytes(data=b, mime_type=m) for (b, m) in images] + [prompt]
        else:
            contents = prompt

        def _call():
            resp = self.client.models.generate_content(
                model=self._cfg.model.name,
                contents=contents,
                config=self._gen_config(system, temp, force_no_think),
            )
            return resp.text

        text = _with_retry(_call, self._cfg.model.max_retries, self._limiter, "gemini.generate")
        # empty/blocked (safety, MAX_TOKENS with no text, etc.): return "" so the
        # caller records an unparseable item rather than retrying forever.
        return text or ""


class ClaudeVertexClient:
    """Claude column. First-party Anthropic API when ANTHROPIC_API_KEY is set
    (model ids like "claude-sonnet-4-6"; a Vertex-style "@date" suffix is
    stripped automatically); otherwise Claude on Vertex AI, as in the paper's
    runs (GOOGLE_CLOUD_PROJECT + application-default credentials)."""

    def __init__(self, cfg: Config):
        import os

        self._cfg = cfg
        self._limiter = _RateLimiter(cfg.model.min_interval_s)
        if os.environ.get("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic

            self.client = Anthropic()
            self._cfg.model.name = self._cfg.model.name.split("@")[0]
        else:
            from anthropic import AnthropicVertex

            self.client = AnthropicVertex(region=cfg.vertex.location, project_id=cfg.vertex.project)

    MAX_OUT = 64000  # claude-sonnet-4-6 output ceiling; larger requests 400

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
                 images: Optional[list] = None, prefix: Optional[str] = None,
                 force_no_think: bool = False) -> str:
        """prefix: shared prompt prefix marked with cache_control (ephemeral) -- calls that share
        it within the TTL pay ~0.1x input on the cached part.
        force_no_think: recovery-ladder calls disable adaptive thinking in the thinking-ON arm."""
        temp = self._cfg.model.temperature if temperature is None else temperature

        def _call():
            if images:
                # multimodal (mmmu_pro): images arrive as raw bytes (same contract as GeminiClient);
                # Anthropic wants base64 blocks.
                import base64

                def mt(b):  # sniff magic bytes; mmmu_pro mixes JPEG/PNG/WebP
                    if b[:3] == b"\xff\xd8\xff": return "image/jpeg"
                    if b[:4] == b"\x89PNG": return "image/png"
                    if b[8:12] == b"WEBP": return "image/webp"
                    if b[:6] in (b"GIF87a", b"GIF89a"): return "image/gif"
                    return "image/png"
                def fit_5mb(b, label=""):
                    # Anthropic hard limit: 5 MB/image (base64). Claude's vision stack downscales to
                    # ~1568px long-edge SERVER-side anyway, so client-side resize to that cap loses no
                    # information the model would ever see. Logged fallback (same policy as fit_context).
                    if len(b) <= 3_700_000:  # API limit applies to BASE64 size (x4/3): raw 3.7MB -> b64 ~4.9MB
                        return b, None
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(b)); img.load()
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    w, h = img.size
                    if max(w, h) > 1568:
                        r = 1568 / max(w, h)
                        img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
                    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
                    out, m2 = buf.getvalue(), "image/png"
                    for q in (92, 85, 75):  # PNG of photos can still exceed the cap -> JPEG ladder
                        if len(out) <= 3_700_000:
                            break
                        buf = io.BytesIO(); img.save(buf, "JPEG", quality=q)
                        out, m2 = buf.getvalue(), "image/jpeg"
                    print(f"[claude-img] {label} {len(b)}B > 5MB cap -> {img.size} {m2} {len(out)}B", flush=True)
                    return out, m2

                content = []
                for im in images:
                    b, m = (im if isinstance(im, tuple) else (im, None))  # mmmu passes (bytes, mime)
                    b, m2 = fit_5mb(b)
                    content.append({"type": "image",
                                    "source": {"type": "base64", "media_type": m2 or m or mt(b),
                                               "data": base64.b64encode(b).decode()}})
                content.append({"type": "text", "text": prompt})
            elif prefix:
                content = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                           {"type": "text", "text": prompt}]
            else:
                content = prompt
            # STREAMING is mandatory: the SDK refuses non-streaming create() at large max_tokens
            # ("Streaming is required for operations that may take longer than 10 minutes").
            # Accumulate and return the final message.
            import os as _os
            _kw = {}
            if _os.environ.get("XCONF_THINKING", "off") == "on" and not force_no_think:
                _kw["thinking"] = {"type": "adaptive"}  # thinking-ON ablation (claude 4.6 adaptive)
                _kw.pop("temperature", None)
            no_temp = "thinking" in _kw
            with self.client.messages.stream(
                model=self._cfg.model.name,
                system=system or "",
                messages=[{"role": "user", "content": content}],
                **({} if no_temp else {"temperature": temp}),
                max_tokens=min(self._cfg.model.max_output_tokens, self.MAX_OUT),
                **_kw,
            ) as stream:
                msg = stream.get_final_message()
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text") if msg.content else ""

        text = _with_retry(_call, self._cfg.model.max_retries, self._limiter, "claude.generate")
        return text or ""


class OpenAICompatClient:
    """Open-weight models served by a LOCAL vLLM OpenAI-compatible server (the GPU experiments).

    Cross-model generality: the method is model-agnostic, so we re-run the whole pipeline on open
    models (Qwen, ...). The model-under-test is local; the retrieval-key EMBEDDER stays on
    Vertex (embedder != model-under-test), so only this client changes. Server URL via env
    VLLM_BASE_URL (default localhost:8000). Multimodal via OpenAI image_url parts (Qwen-VL).
    Thinking disabled (enable_thinking=False) to match the gemini-2.5-flash thinking-off setup.
    """

    def __init__(self, cfg: Config):
        import os
        from openai import OpenAI

        import httpx
        self._cfg = cfg
        self._limiter = _RateLimiter(0.0)  # local vLLM: NO client-side throttle (saturate the server)
        base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        # lift httpx pool limits (default ~caps concurrent connections) so many worker threads
        # actually fire in parallel and saturate the GPU server
        n = int(os.environ.get("XCONF_WORKERS", "24")) + 16
        self.client = OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=600.0,
                             http_client=httpx.Client(limits=httpx.Limits(
                                 max_connections=n, max_keepalive_connections=n)))

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
                 images: Optional[list] = None, prefix: Optional[str] = None,
                 force_no_think: bool = False) -> str:
        import base64, os
        temp = self._cfg.model.temperature if temperature is None else temperature
        if prefix:  # vLLM has automatic prefix caching; just prepend
            prompt = prefix + "\n" + prompt
        think = os.environ.get("XCONF_THINK", "0") == "1"
        if images:
            content = []
            for (b, m) in images:
                uri = f"data:{m};base64," + base64.b64encode(b).decode()
                content.append({"type": "image_url", "image_url": {"url": uri}})
            content.append({"type": "text", "text": prompt})
            user_msg = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": prompt}
        messages = ([{"role": "system", "content": system}] if system else []) + [user_msg]

        # Context-window budgeting (local models have SMALL windows vs Gemini's 1M): clamp max_tokens
        # so prompt + completion fits. Conservative 2 chars/token estimate (LaTeX/chemistry-heavy text
        # really tokenizes at ~2 chars/token) — long salvage-replay prompts get a smaller completion
        # budget instead of a 400 error; short prompts keep the full configured cap.
        ctx = int(os.environ.get("XCONF_CTX", "32768"))
        est_prompt = (len(prompt) + (len(system) if system else 0)) // 2 + 64
        max_out = [min(self._cfg.model.max_output_tokens, max(1024, ctx - est_prompt - 256))]

        def _call():
            resp = self.client.chat.completions.create(
                model=self._cfg.model.name, messages=messages, temperature=temp,
                max_tokens=max_out[0],
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            return resp.choices[0].message.content if resp.choices else ""

        try:
            text = _with_retry(_call, self._cfg.model.max_retries, self._limiter, "vllm.generate")
        except LLMError as e:
            # belt-and-braces: if the estimate was still short, parse the server's exact count and
            # retry once with a precisely clamped completion budget
            import re as _re
            m = _re.search(r"(\d+) in the messages", str(e))
            if m and "maximum context length" in str(e):
                real_prompt = int(m.group(1))
                room = ctx - real_prompt - 128
                if room >= 256:
                    max_out[0] = room
                    text = _with_retry(_call, 2, self._limiter, "vllm.generate.clamped")
                else:
                    raise  # prompt alone (nearly) fills the window; caller's fit_context should shrink it
            else:
                raise
        return text or ""


def build_client(cfg: Config):
    name = cfg.model.name.lower()
    if name.startswith("gemini"):
        return GeminiClient(cfg)
    if name.startswith("claude"):
        return ClaudeVertexClient(cfg)
    return OpenAICompatClient(cfg)  # local vLLM-served open model (Qwen/...)
