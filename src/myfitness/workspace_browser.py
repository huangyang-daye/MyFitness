"""Read-only, project-root-confined file browsing for the local Agent UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

TEXT_EXTENSIONS = {
    ".css",
    ".csv",
    ".env.example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_NAMES = {
    ".chatHistory",
    ".cursor",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".workbuddy",
    "__pycache__",
    "node_modules",
    "skills",
}
SENSITIVE_NAMES = {".env", ".env.local", ".env.production"}
MAX_READ_BYTES = 1_000_000


class WorkspacePathError(ValueError):
    pass


class WorkspaceBrowser:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def list_files(self, relative_path: str = "") -> dict[str, Any]:
        target = self._resolve(relative_path)
        if not target.is_dir():
            raise WorkspacePathError("路径不是目录")
        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if self._excluded(child):
                continue
            item = {
                "name": child.name,
                "path": child.relative_to(self.project_root).as_posix(),
                "type": "directory" if child.is_dir() else "file",
            }
            if child.is_file():
                try:
                    item["size"] = child.stat().st_size
                except OSError:
                    item["size"] = 0
            entries.append(item)
        current = "" if target == self.project_root else target.relative_to(self.project_root).as_posix()
        parent = None if not current else Path(current).parent.as_posix()
        if parent == ".":
            parent = ""
        return {"path": current, "parent": parent, "entries": entries}

    def read_file(self, relative_path: str) -> dict[str, Any]:
        target = self._resolve(relative_path)
        if not target.is_file() or self._excluded(target):
            raise WorkspacePathError("文件不存在或不可读取")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise WorkspacePathError("文件超过 1 MB，未在侧栏中加载")
        if not self._is_text_file(target):
            raise WorkspacePathError("侧栏仅预览文本文件")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspacePathError("文件不是 UTF-8 文本") from exc
        return {
            "path": target.relative_to(self.project_root).as_posix(),
            "name": target.name,
            "content": content,
            "language": self._language_for(target),
            "size": size,
        }

    def search(self, query: str, *, limit: int = 80) -> dict[str, Any]:
        needle = query.strip()
        if len(needle) < 2:
            return {"query": needle, "results": [], "truncated": False}
        lowered = needle.casefold()
        results: list[dict[str, Any]] = []
        truncated = False
        for directory, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [name for name in dirnames if name not in EXCLUDED_NAMES]
            for filename in filenames:
                if len(results) >= limit:
                    truncated = True
                    break
                path = Path(directory) / filename
                if self._excluded(path) or not self._is_text_file(path):
                    continue
                try:
                    if path.stat().st_size > MAX_READ_BYTES:
                        continue
                    for line_number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if lowered in line.casefold():
                            results.append(
                                {
                                    "path": path.relative_to(self.project_root).as_posix(),
                                    "line": line_number,
                                    "excerpt": line.strip()[:240],
                                }
                            )
                            if len(results) >= limit:
                                truncated = True
                                break
                except (OSError, UnicodeDecodeError):
                    continue
            if truncated:
                break
        return {"query": needle, "results": results, "truncated": truncated}

    def _resolve(self, relative_path: str) -> Path:
        raw = str(relative_path or "").replace("\\", "/").lstrip("/")
        target = (self.project_root / raw).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise WorkspacePathError("路径必须位于项目目录内") from exc
        if any(part in EXCLUDED_NAMES or part in SENSITIVE_NAMES for part in target.parts):
            raise WorkspacePathError("该路径不允许在侧栏中访问")
        return target

    def _excluded(self, path: Path) -> bool:
        try:
            relative_parts = path.relative_to(self.project_root).parts
        except ValueError:
            return True
        return any(part in EXCLUDED_NAMES or part in SENSITIVE_NAMES for part in relative_parts)

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        return path.name == ".env.example" or path.suffix.lower() in TEXT_EXTENSIONS

    @staticmethod
    def _language_for(path: Path) -> str:
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
        }.get(path.suffix.lower(), "text")
