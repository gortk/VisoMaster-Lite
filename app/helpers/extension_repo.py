"""
Extension repository access for the Extension Manager.

A repository is a GitHub repo whose ``extensions`` folder (configurable) contains one
sub-folder per extension, each holding the extension's ``.py`` file(s) and a readme.
Single ``.py`` files placed directly in that folder are also treated as extensions.

To stay clear of GitHub's anonymous API rate limit (60 requests/hour per IP), the whole
repository is fetched as ONE zip archive from codeload.github.com and read in memory.
Nothing from the archive is extracted to disk except the ``.py`` files of extensions the
user explicitly chooses to install.

This module has no Qt dependency so it can be unit-tested on its own.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO_URL = (
    "https://github.com/gortk/Visomaster-Lite-Extensions-Repo/tree/main/extensions"
)
DEFAULT_SUBPATH = "extensions"

CONFIG_FILENAME = "_extension_manager.json"

MAX_ZIP_BYTES = 50 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 25

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REF_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_README_NAMES = {"readme.md", "readme.txt", "readme", "readme.markdown"}

STATE_NOT_INSTALLED = "not_installed"
STATE_INSTALLED = "installed"
STATE_UPDATE = "update"


class ExtensionRepoError(Exception):
    """Raised for any user-presentable repository / install problem."""


# --------------------------------------------------------------------------- repo spec
@dataclass(frozen=True)
class RepoSpec:
    owner: str
    repo: str
    ref: str = "HEAD"  # HEAD = the repository's default branch
    subpath: str = DEFAULT_SUBPATH

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.owner}/{self.repo}/zip/{self.ref}"

    @property
    def web_url(self) -> str:
        base = f"https://github.com/{self.owner}/{self.repo}"
        if self.ref != "HEAD" or self.subpath:
            ref = "HEAD" if self.ref == "HEAD" else self.ref
            return f"{base}/tree/{ref}/{self.subpath}".rstrip("/")
        return base

    @property
    def label(self) -> str:
        where = f"/{self.subpath}" if self.subpath else ""
        ref = "" if self.ref == "HEAD" else f"@{self.ref}"
        return f"{self.owner}/{self.repo}{ref}{where}"


def parse_repo_url(text: str) -> RepoSpec:
    """
    Accepts, e.g.:
      https://github.com/owner/repo/tree/main/extensions   (branch + folder)
      https://github.com/owner/repo/tree/main               (branch, default folder)
      https://github.com/owner/repo                         (default branch + folder)
      owner/repo
    When no folder is given, ``extensions`` is used. Branch names containing ``/`` are
    not supported in the URL form.
    """
    raw = (text or "").strip()
    if not raw:
        raise ExtensionRepoError("Please enter a repository URL.")

    raw = raw.split("#", 1)[0].split("?", 1)[0].strip().rstrip("/")
    if raw.endswith(".git"):
        raw = raw[:-4]

    lowered = raw.lower()
    if lowered.startswith(("http://", "https://")):
        without_scheme = raw.split("://", 1)[1]
        host, _, path = without_scheme.partition("/")
        if host.lower() not in ("github.com", "www.github.com"):
            raise ExtensionRepoError(
                "Only GitHub repositories are supported "
                "(e.g. https://github.com/owner/repo/tree/main/extensions)."
            )
    elif lowered.startswith("github.com/"):
        path = raw.split("/", 1)[1]
    else:
        path = raw

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ExtensionRepoError(
            "Could not read the repository. Use a URL like "
            "https://github.com/owner/repo or https://github.com/owner/repo/tree/main/extensions."
        )

    owner, repo = parts[0], parts[1]
    ref = "HEAD"
    subpath = DEFAULT_SUBPATH

    if len(parts) > 2:
        if parts[2] not in ("tree", "blob") or len(parts) < 4:
            raise ExtensionRepoError(
                "Unsupported repository URL. Use .../tree/<branch>/<folder>."
            )
        ref = parts[3]
        rest = parts[4:]
        subpath = "/".join(rest) if rest else DEFAULT_SUBPATH

    if not (_NAME_RE.match(owner) and _NAME_RE.match(repo)):
        raise ExtensionRepoError("Invalid repository owner or name.")
    if ref != "HEAD" and not _REF_RE.match(ref):
        raise ExtensionRepoError("Invalid branch name.")
    if any(seg in ("..", ".") for seg in subpath.split("/")):
        raise ExtensionRepoError("Invalid folder path.")

    return RepoSpec(owner=owner, repo=repo, ref=ref, subpath=subpath.strip("/"))


# ------------------------------------------------------------------------- extensions
@dataclass
class RepoExtension:
    name: str
    files: dict[str, bytes] = field(default_factory=dict)  # filename -> content (.py)
    readme: str = ""
    description: str = ""

    @property
    def file_names(self) -> list[str]:
        return sorted(self.files)


def _first_paragraph(markdown: str, limit: int = 240) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith("#"):
            if lines:
                break
            continue
        lines.append(stripped)
    text = " ".join(lines)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def download_repo_zip(spec: RepoSpec) -> bytes:
    request = urllib.request.Request(
        spec.archive_url,
        headers={"User-Agent": "VisoMaster-Extension-Manager"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_ZIP_BYTES:
                raise ExtensionRepoError("The repository archive is too large.")
            buffer = io.BytesIO()
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
                if buffer.tell() > MAX_ZIP_BYTES:
                    raise ExtensionRepoError("The repository archive is too large.")
            return buffer.getvalue()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ExtensionRepoError(
                f"Repository or branch not found (or it is private): {spec.label}"
            ) from e
        raise ExtensionRepoError(f"GitHub returned HTTP {e.code}.") from e
    except urllib.error.URLError as e:
        raise ExtensionRepoError(f"Could not reach GitHub: {e.reason}") from e
    except TimeoutError as e:
        raise ExtensionRepoError("Timed out while contacting GitHub.") from e


def list_extensions(zip_bytes: bytes, spec: RepoSpec) -> list[RepoExtension]:
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise ExtensionRepoError("The downloaded repository archive is not valid.")

    found: dict[str, RepoExtension] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if not infos:
            raise ExtensionRepoError("The repository is empty.")

        root = infos[0].filename.split("/", 1)[0]
        prefix = f"{root}/{spec.subpath}/" if spec.subpath else f"{root}/"

        for info in infos:
            if not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix) :]
            parts = rel.split("/")
            if any(p in ("", ".", "..") for p in parts) or len(parts) > 2:
                continue
            if info.file_size > MAX_FILE_BYTES:
                continue

            if len(parts) == 1:
                filename = parts[0]
                if not (filename.endswith(".py") and not filename.startswith("_")):
                    continue
                name = filename[:-3]
            else:
                name, filename = parts
                if name.startswith((".", "_")):
                    continue

            if not _NAME_RE.match(name):
                continue

            ext = found.setdefault(name, RepoExtension(name=name))
            lower = filename.lower()
            if filename.endswith(".py") and not filename.startswith("_"):
                if _NAME_RE.match(filename):
                    ext.files[filename] = archive.read(info)
            elif len(parts) == 2 and lower in _README_NAMES:
                ext.readme = archive.read(info).decode("utf-8", errors="replace")

    result = [e for e in found.values() if e.files]
    for ext in result:
        ext.description = _first_paragraph(ext.readme)
    result.sort(key=lambda e: e.name.lower())
    return result


# ------------------------------------------------------------------ install / uninstall
def _normalized_hash(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def installed_state(ext: RepoExtension, extensions_dir: Path) -> str:
    present = [(extensions_dir / name).is_file() for name in ext.files]
    if not any(present):
        return STATE_NOT_INSTALLED
    if all(present):
        try:
            if all(
                _normalized_hash((extensions_dir / name).read_bytes())
                == _normalized_hash(content)
                for name, content in ext.files.items()
            ):
                return STATE_INSTALLED
        except OSError:
            pass
    return STATE_UPDATE


def install_extension(ext: RepoExtension, extensions_dir: Path) -> list[Path]:
    """Validates and writes the extension's .py files. Nothing is written if any file is invalid."""
    prepared: list[tuple[str, bytes]] = []
    for filename, content in ext.files.items():
        if Path(filename).name != filename or not filename.endswith(".py"):
            raise ExtensionRepoError(f"Refusing unsafe file name: {filename}")
        try:
            compile(content.decode("utf-8"), filename, "exec")
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            raise ExtensionRepoError(
                f"{filename} is not valid Python and was not installed ({e})."
            ) from e
        prepared.append((filename, content))

    extensions_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in prepared:
        target = extensions_dir / filename
        fd, tmp_name = tempfile.mkstemp(dir=extensions_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(tmp_name, target)
        except OSError as e:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise ExtensionRepoError(f"Could not write {filename}: {e}") from e
        written.append(target)
    return written


def uninstall_extension(ext: RepoExtension, extensions_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for filename in ext.files:
        target = extensions_dir / filename
        if target.is_file():
            try:
                target.unlink()
            except OSError as e:
                raise ExtensionRepoError(f"Could not remove {filename}: {e}") from e
            removed.append(target)
    return removed


# ------------------------------------------------------------------------------ config
def load_repo_url(extensions_dir: Path) -> str:
    try:
        data = json.loads((extensions_dir / CONFIG_FILENAME).read_text("utf-8"))
        url = str(data.get("repo_url", "")).strip()
        if url:
            return url
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_REPO_URL


def save_repo_url(extensions_dir: Path, url: str) -> None:
    try:
        extensions_dir.mkdir(parents=True, exist_ok=True)
        (extensions_dir / CONFIG_FILENAME).write_text(
            json.dumps({"repo_url": url}, indent=2), encoding="utf-8"
        )
    except OSError as e:
        print(f"[EXTENSION] Could not save extension repository setting: {e}")
