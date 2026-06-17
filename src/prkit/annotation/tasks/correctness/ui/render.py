"""Pure HTML/KaTeX rendering helpers for the correctness UI.

Kept free of any Streamlit import so the math/markup logic can be unit-tested directly.
Math is rendered client-side with KaTeX. When the vendored assets in ``vendor/katex/``
are present (the default), the CSS (with fonts base64-inlined), ``katex.min.js`` and
``auto-render.min.js`` are embedded directly into the HTML so rendering works fully
offline; otherwise it falls back to a CDN. Dollar delimiters are rewritten to
``\\(..\\)`` / ``\\[..\\]`` which KaTeX parses unambiguously.
"""

from __future__ import annotations

import base64
import functools
import html
import mimetypes
import re
from pathlib import Path
from typing import Any

_KATEX_VERSION = "0.16.11"
_CDN = f"https://cdn.jsdelivr.net/npm/katex@{_KATEX_VERSION}/dist"

# Vendored offline assets (see pyproject package-data). May be absent in trimmed installs.
_KATEX_DIR = Path(__file__).resolve().parent / "vendor" / "katex"
_CSS_URL_RE = re.compile(r"url\((['\"]?)([^)'\"]+)\1\)")


def _assets_available() -> bool:
    """True if the vendored KaTeX CSS + JS files are all present on disk."""
    return all(
        (_KATEX_DIR / name).is_file()
        for name in ("katex.min.css", "katex.min.js", "auto-render.min.js")
    )


def _file_to_data_url(path: Path) -> str:
    """Encode *path* as a ``data:`` URI (used to inline KaTeX font files into CSS)."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = {
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
        }.get(path.suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@functools.lru_cache(maxsize=1)
def _inline_katex_css() -> str:
    """KaTeX CSS with every ``url(...)`` font reference rewritten to a base64 data URI."""
    css_path = _KATEX_DIR / "katex.min.css"
    css_text = css_path.read_text(encoding="utf-8")

    def _replace(match: re.Match[str]) -> str:
        asset = match.group(2).strip()
        if asset.startswith("data:"):
            return match.group(0)
        resolved = (css_path.parent / asset).resolve()
        if not resolved.is_file():
            return match.group(0)
        return f"url({_file_to_data_url(resolved)})"

    return _CSS_URL_RE.sub(_replace, css_text)


@functools.lru_cache(maxsize=1)
def _offline_head() -> tuple[str, str]:
    """Return ``(css_html, loader_scripts_html)`` with KaTeX fully inlined (offline).

    Built once per process — base64-inlining ~1MB of fonts is expensive. ``</style>`` /
    ``</script>`` inside the assets are escaped so they cannot terminate the host tags.
    """
    css = _inline_katex_css().replace("</style>", "<\\/style>")
    # Escape any literal "</script>" so it cannot terminate the host <script> tag.
    katex_js = (_KATEX_DIR / "katex.min.js").read_text(encoding="utf-8")
    katex_js = katex_js.replace("</script>", "<\\/script>")
    auto_js = (_KATEX_DIR / "auto-render.min.js").read_text(encoding="utf-8")
    auto_js = auto_js.replace("</script>", "<\\/script>")
    css_html = f"<style>{css}</style>"
    scripts = f"<script>{katex_js}</script><script>{auto_js}</script>"
    return css_html, scripts


def as_text(value: Any) -> str:
    """Coerce *value* to a display string (``None`` -> empty)."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def image_paths_from_data(data: dict[str, Any]) -> list[Path]:
    """Extract image paths from a problem record (``image_path`` str or list)."""
    raw = data.get("image_path")
    out: list[Path] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(Path(item))
    elif isinstance(raw, str) and raw.strip():
        out.append(Path(raw))
    return out


def latex_for_render_zone(text: str) -> str:
    r"""Convert ``$...$`` / ``$$...$$`` to ``\(...\)`` / ``\[...\]`` for KaTeX auto-render.

    Dollar delimiters often mis-parse superscripts (e.g. ``$4^{\circ}$``); the explicit
    ``\( \)`` / ``\[ \]`` forms are unambiguous. Unterminated delimiters are left as-is.
    """
    s = text or ""
    parts: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if i + 1 < n and s[i : i + 2] == "$$":
            j = s.find("$$", i + 2)
            if j < 0:
                parts.append(s[i:])
                break
            parts.append("\\[" + s[i + 2 : j] + "\\]")
            i = j + 2
        elif s[i] == "$":
            j = s.find("$", i + 1)
            if j < 0:
                parts.append(s[i:])
                break
            parts.append("\\(" + s[i + 1 : j] + "\\)")
            i = j + 1
        else:
            parts.append(s[i])
            i += 1
    return "".join(parts)


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def katex_wrap_with_script(inner: str, root_id: str) -> str:
    """Wrap *inner* HTML with KaTeX CSS/JS and an auto-render bootstrap for *root_id*.

    Uses the vendored offline assets when present; otherwise loads KaTeX from the CDN.
    """
    base_style = (
        "<style>html,body{margin:0;padding:0;background:#fff;}"
        f"#{root_id}{{margin:0;padding-bottom:2px;}}</style>"
    )
    if _assets_available():
        katex_css, loader = _offline_head()
    else:
        katex_css = f'<link rel="stylesheet" href="{_CDN}/katex.min.css" crossorigin="anonymous">'
        loader = (
            f'<script defer src="{_CDN}/katex.min.js" crossorigin="anonymous"></script>'
            f'<script defer src="{_CDN}/contrib/auto-render.min.js" crossorigin="anonymous"></script>'
        )
    bootstrap = f"""
<script>
  function tryRender() {{
    var root = document.getElementById("{root_id}");
    if (!root || typeof renderMathInElement === "undefined") return false;
    renderMathInElement(root, {{
      delimiters: [
        {{left: "\\\\(", right: "\\\\)", display: false}},
        {{left: "\\\\[", right: "\\\\]", display: true}},
        {{left: "$$", right: "$$", display: true}},
        {{left: "$", right: "$", display: false}}
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false
    }});
    return true;
  }}
  function scheduleRender() {{
    if (tryRender()) return;
    setTimeout(scheduleRender, 50);
  }}
  document.addEventListener("DOMContentLoaded", scheduleRender);
  setTimeout(scheduleRender, 80);
  setTimeout(scheduleRender, 400);
</script>
"""
    return base_style + katex_css + inner + loader + bootstrap


def katex_html_simple(title: str, text: str) -> str:
    """A single titled block with KaTeX rendering (used for captions/notes)."""
    inner = f"""
<div id="katex-root-simple" style="font-size:15px; line-height:1.55;">
  <div style="font-weight:600; margin-bottom:6px; color:#333;">{_esc(title)}</div>
  <div style="white-space:pre-wrap;">{_esc(latex_for_render_zone(text))}</div>
</div>
"""
    return katex_wrap_with_script(inner, root_id="katex-root-simple")


def dual_field_html(title: str, text: str, *, title_color: str = "#333") -> str:
    """One field: KaTeX-rendered text on top, a divider, then the raw string below."""
    t = text or ""
    t_render = latex_for_render_zone(t)
    return f"""
<div style="border:1px solid #ccc; padding:12px; border-radius:8px; background:#fff;">
  <div style="font-weight:600; margin-bottom:8px; color:{title_color};">{_esc(title)}</div>
  <div style="font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">Rendered</div>
  <div class="katex-render-zone" style="font-size:16px; line-height:1.55; color:#111; white-space:pre-wrap; min-height:1.25em;">{_esc(t_render)}</div>
  <hr style="border:none; border-top:1px solid #d4d4d4; margin:12px 0;" />
  <div style="font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">Raw string</div>
  <pre style="margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; line-height:1.45; color:#333; background:#f5f5f5; padding:10px; border-radius:6px; overflow-x:auto; white-space:pre-wrap; word-break:break-word;">{_esc(t)}</pre>
</div>
"""


def katex_html(question: str, ground_truth: str, model_answer: str) -> str:
    """Question (full width) plus ground-truth and model answer side by side."""
    body = f"""
<div id="katex-root" style="font-size:16px; color:#111;">
  <div style="margin-bottom:14px;">
    {dual_field_html("Question", question, title_color="#333")}
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
    {dual_field_html("Ground truth", ground_truth, title_color="#1b5e20")}
    {dual_field_html("Model answer", model_answer, title_color="#0d47a1")}
  </div>
</div>
"""
    return katex_wrap_with_script(body, root_id="katex-root")


def estimate_main_panel_height(
    question: str, ground_truth: str, model_answer: str
) -> int:
    """Heuristic iframe height; Streamlit reserves the full pixel height it is given."""

    def dual_block_height(text: str) -> int:
        t = text or ""
        if not t.strip():
            return 108
        n_lines = max(1, t.count("\n") + 1, (len(t) + 90) // 91)
        n_lines = min(n_lines, 24)
        return int(58 + n_lines * 19 + 14 + n_lines * 13 + 26)

    h_question = dual_block_height(question)
    h_pair = max(dual_block_height(ground_truth), dual_block_height(model_answer))
    blob = f"{question}\n{ground_truth}\n{model_answer}"
    slack = 56 if "$$" in blob or r"\[" in blob else 32
    total = h_question + h_pair + 10 + slack
    return max(260, min(680, int(total)))
