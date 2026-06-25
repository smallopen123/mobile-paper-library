from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_FILES = [
    ".github/workflows/mobile-paper-library.yml",
    ".gitignore",
    ".env.example",
    "README.md",
    "requirements.txt",
    "data/sent_history.json",
    "docs/index.html",
    "scripts/mobile_paper_library.py",
    "scripts/upload_project.py",
]


class GitHub:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        response = self.session.request(method, f"https://api.github.com{path}", timeout=30, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API {method} {path} failed: {response.status_code} {response.text}")
        return response

    def ensure_repo(self, private: bool = True) -> None:
        response = self.session.get(f"https://api.github.com/repos/{self.repo}", timeout=30)
        if response.status_code == 200:
            print(f"Repository exists: {self.repo}")
            return
        if response.status_code != 404:
            raise RuntimeError(f"GitHub API GET repo failed: {response.status_code} {response.text}")
        owner, name = self.repo.split("/", 1)
        user = self.request("GET", "/user").json()
        if user.get("login", "").lower() != owner.lower():
            raise RuntimeError("Automatic creation supports personal repositories only.")
        self.request(
            "POST",
            "/user/repos",
            json={
                "name": name,
                "private": private,
                "auto_init": False,
                "description": "Mobile bilingual paper library for low-altitude economy research.",
            },
        )
        print(f"Created repository: {self.repo}")

    def default_branch(self) -> str:
        data = self.request("GET", f"/repos/{self.repo}").json()
        return data.get("default_branch") or "main"

    def get_file_sha(self, path: str, branch: str) -> str | None:
        response = self.session.get(
            f"https://api.github.com/repos/{self.repo}/contents/{path}",
            params={"ref": branch},
            timeout=30,
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API GET content failed: {response.status_code} {response.text}")
        return response.json().get("sha")

    def put_file(self, path: str, content: bytes, branch: str) -> None:
        payload = {
            "message": f"Add/update {path}",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
        }
        sha = self.get_file_sha(path, branch)
        if sha:
            payload["sha"] = sha
        self.request("PUT", f"/repos/{self.repo}/contents/{path}", json=payload)

    def enable_pages(self, branch: str) -> None:
        payload = {"source": {"branch": branch, "path": "/docs"}}
        response = self.session.post(f"https://api.github.com/repos/{self.repo}/pages", timeout=30, json=payload)
        if response.status_code in (201, 204):
            print("GitHub Pages enabled.")
            return
        if response.status_code == 409:
            self.request("PUT", f"/repos/{self.repo}/pages", json=payload)
            print("GitHub Pages updated.")
            return
        if response.status_code == 422:
            print("GitHub Pages may already be configured or unavailable for this repo. Please verify in Settings -> Pages.")
            return
        raise RuntimeError(f"GitHub Pages API failed: {response.status_code} {response.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/upload mobile paper library project.")
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/name format.")
    parser.add_argument("--public", action="store_true", help="Create repository as public. Defaults to private.")
    parser.add_argument("--enable-pages", action="store_true", help="Enable GitHub Pages from main/docs.")
    args = parser.parse_args()

    token = os.getenv("GITHUB_PAT") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_PAT or GH_TOKEN.")

    github = GitHub(args.repo, token)
    github.ensure_repo(private=not args.public)
    branch = github.default_branch()
    for path in UPLOAD_FILES:
        local_path = ROOT / path
        github.put_file(path, local_path.read_bytes(), branch)
        print(f"Uploaded: {path}")
    if args.enable_pages:
        github.enable_pages(branch)
    print(f"Uploaded {len(UPLOAD_FILES)} files to {args.repo}@{branch}.")


if __name__ == "__main__":
    main()
