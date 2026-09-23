"""Keep each workflow notebook's implementation cells identical to its .py script.

Each numbered notebook is a real, standalone implementation: the module
preamble (docstring/imports/constants) and every top-level function/class are
copied into their own cells, in source order, so the notebook can be read and
understood top to bottom without opening another file. There is no shared
"wrapper" cell that calls back into the script.

That duplication is exactly what let 11 of these notebooks silently drift out
of sync with their scripts before (see git history / project notes). To keep
that from happening again, this file is the *only* place that writes the
implementation cells, and `--check` verifies every notebook still matches its
script byte-for-byte, so a stale notebook fails `11_validate_project.py`
instead of going unnoticed.

Usage:
  python scripts/sync_notebooks.py            # regenerate every notebook
  python scripts/sync_notebooks.py --check     # verify only; exit 1 if stale
  python scripts/sync_notebooks.py 02_build_training_cohort   # one notebook
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# (notebook/script stem, RUN_ARGS used in the "Run this step" cell, one-line hint)
WORKFLOWS = [
    ("00_download_epa_data", ["--datasets", "toxval"], "download the remaining ToxValDB archive"),
    ("00a_import_toxcast_sql", ["--check", "--login-path", "comptox"], "run: 00a_import_toxcast_sql.py"),
    ("00b_download_structures", [], "download structures"),
    ("01_normalize_epa_data", ["--no-extract"], "execute: 01_normalize_epa_data.py"),
    ("01a_export_toxcast_mysql", ["--check"], "export bioactivity"),
    ("01b_prepare_structures", [], "prepare structures"),
    ("02_build_training_cohort", ["--check"], "execute: 02_build_training_cohort.py"),
    ("03_train_foundation_model", ["--check"], "execute: 03_train_foundation_model.py"),
    ("04_export_embeddings", ["--check"], "execute: 04_export_embeddings.py"),
    ("05_evaluate_foundation_model", ["plan"], "execute: 05_evaluate_foundation_model.py"),
    ("06_prepare_mea_data", [], "execute: 06_prepare_mea_data.py"),
    ("07_query_chemical", ["--chemical", "Permethrin", "--no-pubchem"], "execute: 07_query_chemical.py"),
    ("08_build_knowledge_graph", [], "execute: 08_build_knowledge_graph.py"),
    ("09_register_toxicity_endpoints", ["--check"], "execute: 09_register_toxicity_endpoints.py"),
    ("10_transfer_learning", ["--show-plan"], "execute: 10_transfer_learning.py"),
    ("11_validate_project", [], "execute: 11_validate_project.py"),
]


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def split_script(source: str):
    """Return [(kind, name, code_text), ...] covering the whole script except
    the trailing `if __name__ == "__main__":` guard. kind is 'code' (module
    preamble / statements between functions) or 'def' (one function/class).

    Every line up to the guard (or end of file) is assigned to exactly one
    segment: a segment's end is defined as "right before the next segment
    starts" and the first segment's start is forced to line 1, so blank
    lines and standalone comments between statements are never dropped --
    they ride along with the segment before them."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    guard = next((n for n in tree.body if _is_main_guard(n)), None)
    limit_line = (guard.lineno - 1) if guard else len(lines)
    nodes = [n for n in tree.body if not _is_main_guard(n)]

    merged = []  # [kind, name, start_line, end_line] (1-indexed, inclusive)
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            merged.append(["def", n.name, n.lineno, n.end_lineno])
        elif merged and merged[-1][0] == "code":
            merged[-1][3] = n.end_lineno
        else:
            merged.append(["code", None, n.lineno, n.end_lineno])

    for i in range(len(merged) - 1):
        merged[i][3] = merged[i + 1][2] - 1
    if merged:
        merged[0][2] = 1
        merged[-1][3] = limit_line

    segments = [(kind, name, "".join(lines[start - 1 : end])) for kind, name, start, end in merged if end >= start]

    rebuilt = "".join(seg[2] for seg in segments)
    expected = "".join(lines[:limit_line])
    if rebuilt != expected:
        raise AssertionError("split_script lost or reordered source text; refusing to write a lossy notebook.")
    return segments


def _cell(cell_type: str, source: str, cell_id: str) -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines, "id": cell_id}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _docstring_of(def_source: str) -> str | None:
    """First line of the function/class docstring in this standalone segment, if any."""
    try:
        node = ast.parse(def_source).body[0]
    except (SyntaxError, IndexError):
        return None
    doc = ast.get_docstring(node, clean=True)
    return doc.strip().splitlines()[0] if doc else None


def build_implementation_cells(script_name: str, segments):
    cells = [_cell("markdown", "## 2. Implementation\n\nEvery function below is copied verbatim from "
                    f"`scripts/{script_name}`, in source order, kept in sync by "
                    "`python scripts/sync_notebooks.py` (and checked by `11_validate_project.py`). "
                    "Edit the `.py` file, then re-run the sync script -- never hand-edit these cells.",
                    f"impl-intro-{script_name}")]
    for i, (kind, name, text) in enumerate(segments):
        if kind == "def":
            doc = _docstring_of(text)
            header = f"#### `{name}`" + (f"\n\n{doc}" if doc else "")
            cells.append(_cell("markdown", header, f"impl-md-{script_name}-{i}"))
        cells.append(_cell("code", text.rstrip("\n") + "\n", f"impl-code-{script_name}-{i}"))
    return cells


def build_run_cell(script_name: str, run_args, hint: str) -> dict:
    args_literal = json.dumps(run_args)
    source = (
        "# Set RUN_STEP=True when you are ready to execute this workflow.\n"
        "# The notebook defaults to False so \"Run All\" is safe and does not accidentally\n"
        "# download large files, start a long training job, or overwrite project outputs.\n"
        "RUN_STEP = False\n"
        "\n"
        "# Command-line arguments used when RUN_STEP=True.\n"
        f"RUN_ARGS = {args_literal}\n"
        "\n"
        "if RUN_STEP:\n"
        "    old = sys.argv[:]\n"
        "    try:\n"
        f"        sys.argv = ['{script_name}.py'] + RUN_ARGS\n"
        "        try:\n"
        "            main()\n"
        "        except SystemExit as exc:\n"
        "            # main() uses SystemExit(0) as a CLI success signal (e.g. --check).\n"
        "            # A terminal treats that as silent success; Jupyter displays *any*\n"
        "            # SystemExit as an error-looking traceback, so only re-raise on an\n"
        "            # actual failure (nonzero/non-None exit code).\n"
        "            if exc.code not in (0, None):\n"
        "                raise\n"
        "    finally:\n"
        "        sys.argv = old\n"
        "else:\n"
        "    print('Implementation loaded successfully.')\n"
        f"    print(\"Set RUN_STEP = True in this cell to {hint}\")\n"
        "    print('RUN_ARGS =', RUN_ARGS)\n"
    )
    return _cell("code", source, f"run-{script_name}")


def load_notebook(script_name: str) -> dict:
    path = NOTEBOOKS_DIR / f"{script_name}.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_cell_index(cells) -> int:
    """Index of the notebook's own path-setup code cell (defines PROJECT_ROOT).
    Every notebook family in this project has exactly one, regardless of
    whether the rest of the notebook used the old wrapper pattern or a
    previous copy of the implementation -- so this is the one stable anchor
    to split "head" (title + setup, always kept) from "everything this tool
    owns and regenerates"."""
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "PROJECT_ROOT" in "".join(c["source"]):
            return i
    raise ValueError("Could not find the project-setup cell (no code cell defines PROJECT_ROOT).")


def rebuilt_cells(script_name: str, run_args, hint: str, existing_cells):
    """Keep the notebook's title + project-setup cells untouched; replace
    everything after that with a freshly generated implementation + run
    section, regardless of what was there before (old wrapper, stale copy,
    or already-current)."""
    script_source = (SCRIPTS_DIR / f"{script_name}.py").read_text(encoding="utf-8")
    segments = split_script(script_source)
    head = existing_cells[: _setup_cell_index(existing_cells) + 1]
    return head + build_implementation_cells(script_name, segments) + [build_run_cell(script_name, run_args, hint)]


def sync_one(script_name: str, run_args, hint: str) -> bool:
    """Returns True if the notebook file changed."""
    nb = load_notebook(script_name)
    new_cells = rebuilt_cells(script_name, run_args, hint, nb["cells"])
    changed = new_cells != nb["cells"]
    nb["cells"] = new_cells
    path = NOTEBOOKS_DIR / f"{script_name}.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return changed


def check_one(script_name: str) -> list[str]:
    """Returns a list of human-readable problems (empty if in sync)."""
    script_source = (SCRIPTS_DIR / f"{script_name}.py").read_text(encoding="utf-8")
    segments = split_script(script_source)
    expected_impl = build_implementation_cells(script_name, segments)
    nb = load_notebook(script_name)
    cells = nb["cells"]
    problems = []
    try:
        impl_start = _setup_cell_index(cells) + 1
    except ValueError as exc:
        return [f"{script_name}.ipynb: {exc}"]
    actual_impl = cells[impl_start : impl_start + len(expected_impl)]
    actual_sources = ["".join(c["source"]) for c in actual_impl]
    expected_sources = ["".join(c["source"]) for c in expected_impl]
    if actual_sources != expected_sources:
        problems.append(f"{script_name}.ipynb implementation cells are out of sync with scripts/{script_name}.py")
    run_cell = cells[impl_start + len(expected_impl)] if len(cells) > impl_start + len(expected_impl) else None
    if run_cell is None or "main()" not in "".join(run_cell["source"]):
        problems.append(f"{script_name}.ipynb is missing its run cell (or it no longer calls main())")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Specific script stems to sync/check (default: all)")
    parser.add_argument("--check", action="store_true", help="Verify only; exit 1 if any notebook is stale")
    args = parser.parse_args()

    selected = [w for w in WORKFLOWS if not args.names or w[0] in args.names]
    if not selected:
        raise SystemExit(f"No matching workflow in {[w[0] for w in WORKFLOWS]}")

    if args.check:
        problems = []
        for name, _run_args, _hint in selected:
            problems.extend(check_one(name))
        if problems:
            for p in problems:
                print("STALE:", p)
            print(f"\n{len(problems)} problem(s). Run: python scripts/sync_notebooks.py")
            raise SystemExit(1)
        print(f"Notebook/script sync: PASS ({len(selected)} notebooks match their scripts)")
        return

    changed = 0
    for name, run_args, hint in selected:
        if sync_one(name, run_args, hint):
            changed += 1
            print("updated:", f"notebooks/{name}.ipynb")
        else:
            print("already in sync:", f"notebooks/{name}.ipynb")
    print(f"\n{changed} notebook(s) updated, {len(selected) - changed} already in sync.")


if __name__ == "__main__":
    main()
