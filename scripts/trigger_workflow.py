from __future__ import annotations

import argparse
import os
import time

import requests


WORKFLOW_FILE = "mobile-paper-library.yml"


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

    def default_branch(self) -> str:
        return self.request("GET", f"/repos/{self.repo}").json().get("default_branch") or "main"

    def dispatch(self, ref: str) -> None:
        self.request(
            "POST",
            f"/repos/{self.repo}/actions/workflows/{WORKFLOW_FILE}/dispatches",
            json={"ref": ref},
        )

    def latest_run(self) -> dict:
        runs = self.request(
            "GET",
            f"/repos/{self.repo}/actions/workflows/{WORKFLOW_FILE}/runs",
            params={"per_page": 1},
        ).json().get("workflow_runs", [])
        if not runs:
            raise RuntimeError("No workflow runs found yet.")
        return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger Mobile Paper Library workflow.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()

    token = os.getenv("GITHUB_PAT") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_PAT or GH_TOKEN.")

    github = GitHub(args.repo, token)
    ref = github.default_branch()
    github.dispatch(ref)
    print(f"Dispatched {WORKFLOW_FILE} on {args.repo}@{ref}.")
    if args.wait <= 0:
        return

    deadline = time.time() + args.wait
    time.sleep(8)
    last_status = ""
    while time.time() < deadline:
        run = github.latest_run()
        status = f"{run.get('status')} / {run.get('conclusion')}"
        if status != last_status:
            print(f"Run status: {status}")
            print(f"Run URL: {run.get('html_url')}")
            last_status = status
        if run.get("status") == "completed":
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
