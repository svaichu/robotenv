"""The status table of docs/robots/maniskill.md is the one in tests/maniskill_expected.json (no library needed)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "robots" / "maniskill.md"
ROW = re.compile(r"^\| `([^`]+)` \| ([^|]+) \| (yes|no|-) \|$", re.MULTILINE)
NOT_DOWNLOADED = "refused: not downloaded"


def expected():
    return json.loads((ROOT / "tests" / "maniskill_expected.json").read_text())


def test_the_docs_table_lists_the_expected_status_of_every_agent():
    holds = {"yes": True, "no": False, "-": None}
    rows = {name: {"status": status.strip(), "holds": holds[h]} for name, status, h in ROW.findall(PAGE.read_text())}
    assert rows == expected(), "docs/robots/maniskill.md and tests/maniskill_expected.json disagree"


def test_the_counts_in_the_prose_are_the_counts_of_the_file():
    exp = expected()
    ok = sum(v["status"] == "ok" for v in exp.values())
    missing = sum(v["status"] == NOT_DOWNLOADED for v in exp.values())
    text = PAGE.read_text()
    assert f"Today {ok} run,\n{missing} are not downloaded and {len(exp) - ok - missing} are refused for another reason." in text
    assert f"{sum(v['holds'] is True for v in exp.values())} do." in text and f"These {missing} robots are **untested**" in text


def test_the_scope_statement_is_where_a_reader_looks():
    """ManiSkill is a robot provider only for now: the page, the README, AGENTS.md and the module say so."""
    sources = [PAGE, ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "src" / "robotenv" / "robot" / "providers" / "maniskill" / "__init__.py"]
    for path in sources:
        assert re.search(r"robot provider only,? for now", path.read_text()), path
