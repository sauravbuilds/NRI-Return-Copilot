"""Guards for publishing the repository: no secrets in tracked files, and .env stays ignored."""
import os
import re
from pathlib import Path

import config

ROOT = config.ROOT
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "hf_cache", "chroma", "archive"}
SKIP_FILES = {".env"}  # the real, git-ignored key file
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".csv", ".txt", ".toml", ".ini", ".example", ".cfg"}
SECRET_PATTERNS = {
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "Tavily key": r"tvly-[A-Za-z0-9_-]{16,}",
    "OpenRouter key": r"sk-or-v1-[A-Za-z0-9]{20,}",
    "OpenAI-style key": r"sk-[A-Za-z0-9]{32,}",
    "Google key": r"AIza[0-9A-Za-z_-]{30,}",
    "Gemini key": r"AQ\.Ab[0-9A-Za-z_-]{20,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def project_text_files():
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(folder) / name
            if name not in SKIP_FILES and (path.suffix in TEXT_SUFFIXES or name.startswith(".env")):
                yield path


def test_no_api_keys_in_project_files():
    found = []
    for path in project_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        found += [f"{path.relative_to(ROOT)}: {label}" for label, pattern in SECRET_PATTERNS.items()
                  if re.search(pattern, text)]
    assert not found, "possible secrets: " + "; ".join(found)


def test_env_example_lists_names_only():
    lines = [l.strip() for l in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    assert lines and all(l.endswith("=") for l in lines if "=" in l)


def test_gitignore_keeps_secrets_and_generated_data_out():
    raw = (ROOT / ".gitignore").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a byte-order mark can break the first rule"
    rules = {l.strip() for l in raw.decode("utf-8").splitlines()}
    assert {".env", ".env.*", "!.env.example", ".streamlit/secrets.toml", ".venv/",
            "data/chroma/", "data/hf_cache/"} <= rules


def test_workflow_passes_no_secrets_to_the_refresh_job():
    workflow = (ROOT / ".github" / "workflows" / "refresh.yml").read_text(encoding="utf-8")
    assert "secrets." not in workflow


def test_streamlit_hides_error_details_from_visitors():
    settings = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'showErrorDetails = "none"' in settings and "gatherUsageStats = false" in settings
