"""Every constant traces to a source. A parameter with no row fails CI.

This is the test that keeps ``docs/research/`` honest. It is easy to write a research
dossier once and let the code drift away from it; it is not easy to do that with this test
in the suite.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULTS = REPO / "src" / "hcmtwin" / "defaults.py"
PROVENANCE = REPO / "docs" / "research" / "04_model_provenance.md"
RESEARCH = REPO / "docs" / "research"

VALID_CONFIDENCE = {"measured", "calibrated", "assumed"}

NUMERIC_GUARDS_OUTSIDE_DEFAULTS: dict[str, set[str]] = {
    "hcmtwin/sarcomere.py": {"CA50_FLOOR_UM"},
    "hcmtwin/model.py": {"MMHG_ML_TO_JOULE"},
    "hcmtwin/population.py": {"DEFAULT_SEED"},
    "hcmtwin/analysis/identifiability.py": {"CONFOUNDING_THRESHOLD"},
    "hcmtwin/analysis/surrogate.py": {"DEFAULT_DEGREE", "HOLDOUT_FRACTION"},
    "hcmtwin/analysis/tiebreaker.py": {"MIN_SIGNAL_TO_NOISE"},
    "hcmtwin/viz/dashboard.py": {"LOOP_POINTS"},
}
"""Module-level numeric constants allowed to live outside ``defaults.py``.

Each is a numerical guard, an exact unit conversion, a random seed, or an *analysis
convention*. None makes a claim about physiology, which is the line this exemption draws:
a reporting threshold is a choice the reader can disagree with and is stated in the text
next to the result, whereas a physiological constant is a claim about the world and needs
a source. Anything that is a claim about the world belongs in ``defaults.py``."""


def _module_constants(path: pathlib.Path) -> dict[str, float]:
    """Module-level uppercase assignments of numeric literals."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        literal: float | None = None
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            literal = float(value.value)
        elif (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
            and isinstance(value.operand.value, (int, float))
        ):
            literal = -float(value.operand.value)
        if literal is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                found[target.id] = literal
    return found


def _provenance_rows() -> dict[str, dict[str, str]]:
    """Parse the pipe table at the end of the provenance document."""
    if not PROVENANCE.exists():
        pytest.fail(f"missing provenance document: {PROVENANCE}")
    rows: dict[str, dict[str, str]] = {}
    for line in PROVENANCE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 6:
            continue
        # Skip the markdown header separator, which is dashes and colons in every cell.
        if all(cell and set(cell) <= set("-: ") for cell in cells):
            continue
        name = cells[0].strip("`")
        if not name.isupper():
            continue
        rows[name] = {
            "symbol": cells[1],
            "value": cells[2],
            "units": cells[3],
            "source": cells[4],
            "confidence": cells[5].lower(),
        }
    return rows


def test_every_default_has_a_provenance_row() -> None:
    constants = _module_constants(DEFAULTS)
    rows = _provenance_rows()
    missing = sorted(set(constants) - set(rows))
    assert not missing, (
        "constants in defaults.py with no row in 04_model_provenance.md: " + ", ".join(missing)
    )


def test_no_orphan_provenance_rows() -> None:
    constants = _module_constants(DEFAULTS)
    rows = _provenance_rows()
    orphans = sorted(set(rows) - set(constants))
    assert not orphans, "provenance rows for constants that no longer exist: " + ", ".join(orphans)


def test_provenance_values_match_the_code() -> None:
    """A stale table is worse than no table: it looks like a citation and is not one."""
    constants = _module_constants(DEFAULTS)
    rows = _provenance_rows()
    mismatched: list[str] = []
    for name, value in constants.items():
        if name not in rows:
            continue
        recorded = rows[name]["value"].strip("`")
        try:
            parsed = float(recorded)
        except ValueError:
            mismatched.append(f"{name}: value cell {recorded!r} is not a number")
            continue
        if abs(parsed - value) > 1e-9 * max(1.0, abs(value)):
            mismatched.append(f"{name}: code has {value}, table says {parsed}")
    assert not mismatched, "; ".join(mismatched)


def test_confidence_labels_are_valid() -> None:
    bad = {
        name: row["confidence"]
        for name, row in _provenance_rows().items()
        if row["confidence"] not in VALID_CONFIDENCE
    }
    assert not bad, f"confidence must be one of {sorted(VALID_CONFIDENCE)}; got {bad}"


def test_every_provenance_row_cites_something() -> None:
    """A row whose source cell is empty is a missing citation wearing a table."""
    empty = [
        name
        for name, row in _provenance_rows().items()
        if len(row["source"]) < 4 or row["source"] in {"-", "n/a", "TODO"}
    ]
    assert not empty, "provenance rows with no source: " + ", ".join(sorted(empty))


def test_no_stray_physiological_constants_outside_defaults() -> None:
    """All physiological constants live in defaults.py, with a short documented exception list."""
    package = REPO / "src" / "hcmtwin"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        if path.name in {"defaults.py", "units.py"}:
            continue
        relative = str(path.relative_to(REPO / "src"))
        allowed = NUMERIC_GUARDS_OUTSIDE_DEFAULTS.get(relative, set())
        for name in _module_constants(path):
            if name not in allowed:
                offenders.append(f"{relative}:{name}")
    assert not offenders, (
        "numeric constants outside defaults.py with no documented exemption: "
        + ", ".join(offenders)
    )


def test_research_dossier_files_all_exist() -> None:
    expected = [
        "00_overview.md",
        "01_disease_mechanism.md",
        "02_drug_and_dosing.md",
        "03_prior_computational.md",
        "04_model_provenance.md",
        "05_validation_targets.md",
        "06_measurement_noise.md",
        "07_gap_statement.md",
        "bibliography.bib",
    ]
    missing = [name for name in expected if not (RESEARCH / name).exists()]
    assert not missing, f"missing research dossier files: {missing}"


def test_no_unresolved_verify_markers() -> None:
    """``[VERIFY]`` means "I have not checked this against the primary source".

    Definition of done requires zero of them. ``[GAP]`` is allowed and expected: it means
    the literature does not supply what the model needs, which is a finding, not a debt.
    """
    offenders: list[str] = []
    for path in sorted(RESEARCH.glob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "[VERIFY]" in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, "unresolved [VERIFY] markers: " + ", ".join(offenders)


def test_every_gap_marker_is_justified() -> None:
    """A ``[GAP]`` must be followed by prose saying what is missing and what we did.

    Checked per paragraph rather than per line, because the constraint is about content
    and line breaks are just formatting. A marker with nothing after it is a debt dressed
    as a disclosure.
    """
    offenders: list[str] = []
    for path in sorted(RESEARCH.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
            if "[GAP]" not in paragraph:
                continue
            tail = paragraph.split("[GAP]", 1)[1]
            if len(" ".join(tail.split())) < 60:
                offenders.append(f"{path.name}: paragraph {index}")
    assert not offenders, "[GAP] markers with no explanation of what is missing: " + ", ".join(
        offenders
    )
