# tests/test_extraction.py
from pathlib import Path

from ragpipe.extraction import html_to_markdown

FIXTURE = Path(__file__).parent / "fixtures" / "mslearn_sample.html"


def test_extracts_main_content_only():
    md, used_main = html_to_markdown(FIXTURE.read_text())
    assert used_main is True
    assert "Skip to main content" not in md     # nav dropped
    assert "Feedback" not in md                 # footer dropped
    assert "telemetry" not in md                # script dropped


def test_preserves_structure():
    md, _ = html_to_markdown(FIXTURE.read_text())
    assert "# Consistency levels in Azure Cosmos DB" in md
    assert "## Configure the default consistency level" in md
    assert "az cosmosdb update" in md
    assert "```" in md                          # fenced code survives
    assert "|" in md and "Strong" in md         # table survives


def test_falls_back_to_whole_page_without_main():
    md, used_main = html_to_markdown("<html><body><p>plain body</p></body></html>")
    assert used_main is False
    assert "plain body" in md


def test_empty_input():
    md, used_main = html_to_markdown("")
    assert md == ""
    assert used_main is False
