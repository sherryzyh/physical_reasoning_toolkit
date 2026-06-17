"""Tests for the pure rendering helpers of the correctness UI (no Streamlit import)."""

from __future__ import annotations

from prkit.annotation.tasks.correctness.ui import render


def test_as_text():
    assert render.as_text(None) == ""
    assert render.as_text("x") == "x"
    assert render.as_text(5) == "5"


def test_latex_for_render_zone_conversions():
    assert render.latex_for_render_zone("$x$") == "\\(x\\)"
    assert render.latex_for_render_zone("$$y$$") == "\\[y\\]"
    assert render.latex_for_render_zone("a $b$ c") == "a \\(b\\) c"
    # Unterminated delimiter is left untouched.
    assert render.latex_for_render_zone("$x") == "$x"


def test_image_paths_from_data_variants():
    assert render.image_paths_from_data({}) == []
    assert [str(p) for p in render.image_paths_from_data({"image_path": "a.png"})] == [
        "a.png"
    ]
    multi = render.image_paths_from_data({"image_path": ["a.png", "", "b.png"]})
    assert [str(p) for p in multi] == ["a.png", "b.png"]


def test_estimate_main_panel_height_bounds():
    h = render.estimate_main_panel_height("q" * 5000, "gt", "pred")
    assert 260 <= h <= 680


def test_katex_html_contains_fields_and_loader():
    out = render.katex_html("What is $E=mc^2$?", "ground", "model")
    assert "renderMathInElement" in out
    assert "Ground truth" in out and "Model answer" in out


def test_offline_assets_inline_fonts_and_no_cdn():
    # The vendored assets ship with the package, so rendering must be offline.
    assert render._assets_available() is True
    out = render.katex_html("$x$", "gt", "pred")
    assert "data:font/woff2;base64," in out  # fonts inlined as data URIs
    assert "<style>" in out
    assert "cdn.jsdelivr.net" not in out  # no network dependency


def test_file_to_data_url_mime(tmp_path):
    font = tmp_path / "KaTeX_Main-Regular.woff2"
    font.write_bytes(b"\x00\x01\x02")
    assert render._file_to_data_url(font).startswith("data:font/woff2;base64,")


def test_cdn_fallback_when_assets_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_KATEX_DIR", tmp_path / "absent")
    out = render.katex_html("$x$", "gt", "pred")
    assert "cdn.jsdelivr.net" in out
    assert "data:font" not in out
