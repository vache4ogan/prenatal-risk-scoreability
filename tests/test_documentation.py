"""Public documentation links and plotted claims stay tied to the release."""

import csv
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
NEURIPS = "https://neurips.cc/virtual/2026/loc/sydney/178202"


def test_paper_citation_and_main_link():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert NEURIPS in readme
    assert citation["url"] == NEURIPS
    paper = citation["preferred-citation"]
    assert paper["url"] == NEURIPS
    assert paper["authors"] == citation["authors"]
    assert paper["year"] == 2026
    assert "non-archival" in paper["notes"].lower()


@pytest.mark.parametrize("relative", [
    "README.md", "docs/reproduce.md", "pretrained/README.md",
])
def test_local_markdown_links(relative):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        destination = path.parent / unquote(link.path) if link.path else path
        assert destination.exists(), (relative, target)
        if link.fragment and destination.suffix == ".md":
            headings = re.findall(
                r"^#{1,6}\s+(.+)$", destination.read_text(encoding="utf-8"), re.M
            )
            anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
                       for h in headings}
            assert unquote(link.fragment) in anchors, (relative, target)


def test_overview_matches_reference_point_estimates():
    figure = ET.parse(ROOT / "docs/images/equal-budget.svg").getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    with (ROOT / "results/reference/2023/shapley_main_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [r for r in csv.DictReader(handle) if float(r["nominal_fraction"]) == 0.1]
    assert len(rows) == 3
    for row in rows:
        panel = figure.find(f".//s:g[@id='{row['target']}']", ns)
        assert panel is not None
        same = float(row["total_topq_contrast_point_pp"])
        equal = float(row["ordered_availability_at_all_budget_point_pp"])
        assert float(panel.attrib["data-same-fraction-pp"]) == pytest.approx(same)
        assert float(panel.attrib["data-equal-budget-pp"]) == pytest.approx(equal)
        labels = {element.text for element in panel.findall("s:text", ns)}
        assert f"+{same:.2f}" in labels
        assert f"+{equal:.2f}" in labels
        for key, value in [("same-fraction", same), ("equal-budget", equal)]:
            bar = panel.find(f"s:rect[@data-series='{key}']", ns)
            assert float(bar.attrib["width"]) == pytest.approx(value * 62, abs=0.01)
    description = figure.find("s:desc", ns).text
    assert "2023" in description and "larger" in description
