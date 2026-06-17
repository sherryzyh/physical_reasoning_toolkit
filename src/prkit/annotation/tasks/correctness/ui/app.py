#!/usr/bin/env python3
"""Streamlit UI for the ``correctness`` annotation task.

Review each problem's question + ground truth vs. the model answer (with KaTeX math),
then write an integer ``correctness`` (1 = correct, 0 = incorrect) into the
``problem_*.json`` file. Optionally flag a wrong dataset answer with
``is_answer_correct = false``.

Launched by ``prkit annotate correctness <answers_dir>`` (which shells out to
``streamlit run`` and forwards ``--answers-dir`` / ``--annotator`` after ``--``).
Run as a script, so it imports the installed ``prkit`` package by absolute path.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from prkit.annotation.tasks.correctness import store
from prkit.annotation.tasks.correctness.ui import render


def _cli_value(flag: str) -> str | None:
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
    return None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_correctness_and_advance(
    paths: list[Path], idx: int, correctness: int
) -> None:
    """Write ``correctness`` (+ annotator metadata), then advance to the next unlabelled.

    Saving from the last row re-scans the folder so newly arrived files are picked up.
    """
    path = paths[idx]
    data = store.load_json(path)
    data["correctness"] = correctness
    data["correctness_annotator"] = st.session_state.get("annotator", "")
    data["correctness_timestamp"] = _now_iso()
    store.save_json(path, data)

    if idx < len(paths) - 1:
        st.session_state.idx = store.next_unannotated_index(paths, idx + 1)
        st.rerun()
        return

    folder = Path(st.session_state.folder_str).expanduser()
    if folder.is_dir():
        st.session_state.paths = store.list_problem_jsons(folder)
    paths = st.session_state.paths

    if not paths or store.all_files_annotated(paths):
        st.session_state.idx = 0
    else:
        first_i, _ = store.first_unannotated_index(paths)
        st.session_state.idx = first_i
    st.rerun()


def _paint_annotation_buttons_via_parent_iframe() -> None:
    """Colour the main-area buttons by label text via a same-origin parent-document script."""
    snippet = """
<script>
(function () {
  var PAIRS = [
    ["Correct — save & next", "#2e7d32", "#ffffff", "1px solid #1b5e20"],
    ["Incorrect — save & next", "#c62828", "#ffffff", "1px solid #b71c1c"],
    ["Ground truth is incorrect", "#ef6c00", "#ffffff", "1px solid #e65100"],
    ["◀ Previous", "#eceff1", "#263238", "1px solid #b0bec5"],
    ["Next ▶", "#eceff1", "#263238", "1px solid #b0bec5"]
  ];
  function paintSubtreeColor(el, fg) {
    el.style.setProperty("color", fg, "important");
    el.style.setProperty("-webkit-text-fill-color", fg, "important");
    el.style.setProperty("opacity", "1", "important");
    el.querySelectorAll("*").forEach(function (n) {
      var tag = (n.tagName || "").toUpperCase();
      if (tag === "SCRIPT" || tag === "STYLE") return;
      n.style.setProperty("color", fg, "important");
      n.style.setProperty("-webkit-text-fill-color", fg, "important");
      n.style.setProperty("opacity", "1", "important");
      if (tag === "SVG" || tag === "PATH" || tag === "G") {
        n.style.setProperty("fill", fg, "important");
      }
    });
  }
  function paint() {
    try {
      var doc = window.parent && window.parent.document;
      if (!doc) return;
      var root = doc.querySelector('section[data-testid="stMain"]')
        || doc.querySelector('[data-testid="stAppViewContainer"] main')
        || doc.querySelector("main");
      if (!root) return;
      root.querySelectorAll('[data-testid="stButton"] button').forEach(function (btn) {
        var t = (btn.innerText || btn.textContent || "").replace(/\\s+/g, " ").trim();
        if (btn.disabled) {
          btn.style.setProperty("background-color", "#eeeeee", "important");
          btn.style.setProperty("border", "1px solid #e0e0e0", "important");
          paintSubtreeColor(btn, "#757575");
          return;
        }
        for (var i = 0; i < PAIRS.length; i++) {
          if (t.indexOf(PAIRS[i][0]) !== -1) {
            btn.style.setProperty("background-color", PAIRS[i][1], "important");
            btn.style.setProperty("border", PAIRS[i][3], "important");
            btn.style.setProperty("font-weight", "600", "important");
            paintSubtreeColor(btn, PAIRS[i][2]);
            return;
          }
        }
      });
    } catch (e) {}
  }
  paint();
  setTimeout(paint, 0);
  setTimeout(paint, 100);
  setTimeout(paint, 400);
  setTimeout(paint, 1000);
})();
</script>
"""
    components.html(snippet, height=1, scrolling=False)


def _render_sidebar(default_folder: str, default_annotator: str) -> None:
    with st.sidebar:
        st.subheader("Annotator")
        st.text_input("Your name", value=default_annotator, key="annotator")

        st.subheader("Folder")
        folder_in = st.text_input(
            "Model-answers folder (contains problem_*.json)",
            value=st.session_state.folder_str,
            key="folder_input",
            help="e.g. …/response_with_answer_tag/seephys_gpt-5-nano",
        )
        if st.button("Load folder"):
            st.session_state.folder_str = folder_in.strip()
            folder = Path(st.session_state.folder_str).expanduser()
            if not folder.is_dir():
                st.error(f"Not a directory: {folder}")
            else:
                st.session_state.paths = store.list_problem_jsons(folder)
                if not st.session_state.paths:
                    st.warning("No problem_*.json files found.")
                    st.session_state.idx = 0
                else:
                    first_i, all_labelled = store.first_unannotated_index(
                        st.session_state.paths
                    )
                    st.session_state.idx = first_i
                    n = len(st.session_state.paths)
                    if all_labelled:
                        st.success(f"Loaded {n} files — all already labelled.")
                    else:
                        st.success(f"Loaded {n} files (opened first unlabelled).")

        paths: list[Path] = st.session_state.paths
        if paths:
            total = len(paths)
            done = store.count_annotated(paths)
            st.progress(done / total if total else 0.0)
            st.caption(f"Labelled {done} / {total}")
        if render._assets_available():
            st.caption("Math renders offline from vendored KaTeX assets.")
        else:
            st.caption("Math rendering fetches KaTeX from a CDN once per session.")


def main() -> None:
    st.set_page_config(page_title="Correctness annotation", layout="wide")
    st.title("Correctness annotation")

    default_folder = _cli_value("--answers-dir") or ""
    default_annotator = _cli_value("--annotator") or ""

    if "paths" not in st.session_state:
        st.session_state.paths = []
    if "folder_str" not in st.session_state:
        st.session_state.folder_str = default_folder
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    _render_sidebar(default_folder, default_annotator)

    paths = st.session_state.paths
    if not paths:
        st.info("Enter a folder path in the sidebar and click **Load folder**.")
        return

    folder_name = Path(st.session_state.folder_str).name
    if "_" in folder_name:
        dataset, _, model = folder_name.partition("_")
        st.caption(f"Dataset **{dataset}** · model **{model}**")

    if store.all_files_annotated(paths):
        st.success(
            "**Completed** — every `problem_*.json` in this folder has a `correctness` label."
        )
        st.caption(f"{len(paths)} file(s) · `{st.session_state.folder_str}`")
        return

    idx = max(0, min(int(st.session_state.idx), len(paths) - 1))
    st.session_state.idx = idx

    path = paths[idx]
    try:
        data: dict[str, Any] = store.load_json(path)
    except (OSError, ValueError) as error:
        st.error(f"Could not read {path}: {error}")
        return

    question = render.as_text(data.get("question"))
    ground_truth = render.as_text(data.get("answer"))
    model_answer = render.as_text(data.get("model_answer"))
    pid = render.as_text(data.get("problem_id", path.stem))

    st.caption(
        f"File **{path.name}** · problem_id **{pid}** · {idx + 1} / {len(paths)}"
    )

    panel_h = render.estimate_main_panel_height(question, ground_truth, model_answer)
    components.html(
        render.katex_html(question, ground_truth, model_answer),
        height=panel_h,
        scrolling=True,
    )

    cur = data.get("correctness")
    st.markdown(
        "**Label** — writes integer `correctness` (`1` = correct, `0` = incorrect) into this JSON file."
    )
    if "correctness_predicate" in data:
        st.caption(
            f"LLM-judge hint · correctness_predicate = {data.get('correctness_predicate')!r}"
        )

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button(
            "Correct — save & next", key="hc_annot_correct", use_container_width=True
        ):
            _save_correctness_and_advance(paths, idx, 1)
    with bc2:
        if st.button(
            "Incorrect — save & next",
            key="hc_annot_incorrect",
            use_container_width=True,
        ):
            _save_correctness_and_advance(paths, idx, 0)

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button(
            "◀ Previous", key="hc_nav_prev", use_container_width=True, disabled=idx <= 0
        ):
            st.session_state.idx = idx - 1
            st.rerun()
    with nav2:
        if st.button(
            "⚠ Ground truth is incorrect",
            key="hc_nav_gt_wrong",
            use_container_width=True,
            help="Writes is_answer_correct=false in this JSON; stays on this problem.",
        ):
            data["is_answer_correct"] = False
            store.save_json(path, data)
            st.warning("Saved is_answer_correct=false (dataset answer flagged).")
            st.rerun()
    with nav3:
        if st.button(
            "Next ▶",
            key="hc_nav_next",
            use_container_width=True,
            disabled=idx >= len(paths) - 1,
        ):
            st.session_state.idx = store.next_unannotated_index(paths, idx + 1)
            st.rerun()

    _paint_annotation_buttons_via_parent_iframe()

    with st.expander("Optional context"):
        img_paths = render.image_paths_from_data(data)
        existing = [p for p in img_paths if p.is_file()]
        missing = [p for p in img_paths if not p.is_file()]
        if img_paths:
            st.markdown("**Images**")
            for ip in existing:
                st.image(str(ip), use_container_width=False)
            if missing:
                st.caption(
                    "Paths in JSON not found on disk: "
                    + ", ".join(str(p) for p in missing)
                )
        caption = render.as_text(data.get("caption"))
        if caption:
            components.html(
                render.katex_html_simple("Caption", caption), height=220, scrolling=True
            )

    st.divider()
    jump = st.number_input(
        "Jump to index (1-based)",
        min_value=1,
        max_value=len(paths),
        value=idx + 1,
        key="jump_n",
    )
    if st.button("Go", key="jump_go"):
        st.session_state.idx = int(jump) - 1
        st.rerun()

    if cur is not None:
        st.info(
            f"This file already has **correctness** = {cur!r} (overwrite by saving again)."
        )


if __name__ == "__main__":
    main()
