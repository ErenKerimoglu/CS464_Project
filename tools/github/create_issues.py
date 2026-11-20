"""Create the project's labels, milestones and issues on GitHub from docs/issues/*.md.

Usage (from the repo root):
    python tools/github/create_issues.py all [--dry-run]   # labels + milestones + issues
    python tools/github/create_issues.py setup             # labels + milestones only
    python tools/github/create_issues.py issues            # issues only (skips titles that already exist)
    python tools/github/create_issues.py assign            # assign issues using tools/github/team.yaml

Auth: uses $GITHUB_TOKEN if set, otherwise `gh auth token` (GitHub CLI, after `gh auth login`).
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISSUES_DIR = ROOT / "docs" / "issues"
TEAM_FILE = Path(__file__).with_name("team.yaml")
REPO = os.environ.get("GITHUB_REPO", "ErenKerimoglu/CS464_Project")
API = f"https://api.github.com/repos/{REPO}"

OWNER_NAMES = {
    "eren": "Eren Kerimoğlu",
    "ege": "Ege Ural",
    "yunus": "Yunus Dildar Çelik",
    "berkay": "Berkay Ataman",
    "eftelya": "Eftelya Çoban",
    "all": "Everyone",
}

LABELS = {
    # owners
    "owner:eren": ("1d76db", "Owned by Eren Kerimoğlu"),
    "owner:ege": ("0e8a16", "Owned by Ege Ural"),
    "owner:yunus": ("fbca04", "Owned by Yunus Dildar Çelik"),
    "owner:berkay": ("d93f0b", "Owned by Berkay Ataman"),
    "owner:eftelya": ("5319e7", "Owned by Eftelya Çoban"),
    "owner:all": ("bfdadc", "Shared by the whole team"),
    # areas
    "area:data": ("c5def5", "Dataset, dedup, split, preprocessing"),
    "area:infra": ("c5def5", "Training framework, runners, tooling"),
    "area:evaluation": ("c5def5", "Metrics, results, aggregation"),
    "area:classical": ("c5def5", "Hand-crafted features and classical ML"),
    "area:models": ("c5def5", "Deep model architectures and tuning"),
    "area:orientation": ("c5def5", "Rotation invariance / equivariance (main question)"),
    "area:experiments": ("c5def5", "Planned experiments E2-E6"),
    # types
    "type:setup": ("ededed", "Onboarding and setup"),
    "type:feature": ("a2eeef", "Implementation of a module"),
    "type:experiment": ("7057ff", "Running an experiment and reporting results"),
    "type:report": ("0052cc", "Report and presentation"),
    # priority
    "priority:critical": ("b60205", "On the critical path, blocks others"),
    "priority:high": ("d93f0b", "Required for the final report"),
    "priority:stretch": ("cccccc", "Only if time permits"),
}

MILESTONES = {
    "M0 - Foundations": ("2026-11-29T23:59:59Z", "W1: split committed, core contracts implemented with tests"),
    "M1 - Pipelines & Baselines": ("2026-12-06T23:59:59Z", "W2: dataset/GPU preprocessing, runners, statmorph, backbones, escnn smoke run"),
    "M2 - Core Experiments": ("2026-12-20T23:59:59Z", "W3-W4: classical baselines, tuning, family comparison, E1a, E3, E4"),
    "M3 - Analysis": ("2026-12-27T23:59:59Z", "W5: rotation sweep + synthesis, E2, E5, E6, classical orientation, SDSS"),
    "M4 - Report": ("2027-01-03T23:59:59Z", "W6: final report and presentation"),
}


# --------------------------------------------------------------------------- helpers
def token() -> str:
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("No token: set GITHUB_TOKEN or install GitHub CLI and run `gh auth login`.")


def api(method: str, path: str, payload: dict | None = None, tok: str = ""):
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} -> {e.code}: {msg}") from None


def paginate(path: str, tok: str) -> list:
    out, page = [], 1
    while True:
        sep = "&" if "?" in path else "?"
        batch = api("GET", f"{path}{sep}per_page=100&page={page}", tok=tok)
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def parse_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return [value] if value else []
    return [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]


def load_issues() -> list[dict]:
    issues = []
    for f in sorted(ISSUES_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8").replace("\r\n", "\n")
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        if not m:
            sys.exit(f"{f.name}: missing front matter")
        meta: dict = {}
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        issues.append({
            "id": int(meta["id"]),
            "title": meta["title"].strip("'\""),
            "owner": meta["owner"],
            "labels": parse_list(meta.get("labels", "[]")),
            "milestone": meta["milestone"].strip("'\""),
            "week": meta.get("week", ""),
            "depends_on": [int(x) for x in parse_list(meta.get("depends_on", "[]"))],
            "body": m.group(2).strip(),
            "file": f.name,
        })
    issues.sort(key=lambda i: i["id"])
    ids = [i["id"] for i in issues]
    if ids != list(range(1, len(ids) + 1)):
        sys.exit(f"Issue ids must be 1..N without gaps, got {ids}")
    return issues


def load_team() -> dict[str, str]:
    team = {}
    for line in TEAM_FILE.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, _, v = line.partition(":")
            team[k.strip()] = v.strip().strip("'\"")
    return team


def assignees_for(owner: str, team: dict[str, str]) -> list[str]:
    if owner == "all":
        return [u for u in team.values() if u]
    return [team[owner]] if team.get(owner) else []


def render_body(issue: dict) -> str:
    deps = ", ".join(f"#{d}" for d in issue["depends_on"]) or "none"
    header = (
        f"> **Owner:** {OWNER_NAMES[issue['owner']]} · **Target week:** {issue['week']} · "
        f"**Milestone:** {issue['milestone']} · **Depends on:** {deps}\n>\n"
        f"> Specs: [`docs/PROJECT_PLAN.md`](../blob/main/docs/PROJECT_PLAN.md) · "
        f"Source of this issue: [`docs/issues/{issue['file']}`](../blob/main/docs/issues/{issue['file']})\n\n"
    )
    footer = "\n\n---\n**Definition of done:** tasks ticked · tests pass in CI · PR reviewed and merged · results/figures committed where relevant · short summary comment on this issue."
    return header + issue["body"] + footer


# --------------------------------------------------------------------------- commands
def cmd_setup(tok: str, dry: bool) -> None:
    existing = {l["name"] for l in paginate("/labels", tok)} if not dry else set()
    for name, (color, desc) in LABELS.items():
        if name in existing:
            continue
        print(f"label    + {name}")
        if not dry:
            api("POST", "/labels", {"name": name, "color": color, "description": desc}, tok)
    existing_ms = {m["title"] for m in paginate("/milestones?state=all", tok)} if not dry else set()
    for title, (due, desc) in MILESTONES.items():
        if title in existing_ms:
            continue
        print(f"milestone+ {title}")
        if not dry:
            api("POST", "/milestones", {"title": title, "due_on": due, "description": desc}, tok)


def cmd_issues(tok: str, dry: bool) -> None:
    issues, team = load_issues(), load_team()
    if dry:
        for i in issues:
            print(f"#{i['id']:>2} [{i['owner']:<7}] {i['milestone']:<28} {i['title']}")
        return
    ms = {m["title"]: m["number"] for m in paginate("/milestones?state=all", tok)}
    existing = {i["title"]: i["number"] for i in paginate("/issues?state=all", tok) if "pull_request" not in i}
    for i in issues:
        if i["title"] in existing:
            print(f"skip     #{existing[i['title']]} {i['title']} (exists)")
            continue
        payload = {
            "title": i["title"],
            "body": render_body(i),
            "labels": i["labels"] + [f"owner:{i['owner']}"],
            "milestone": ms[i["milestone"]],
        }
        try:
            created = api("POST", "/issues", {**payload, "assignees": assignees_for(i["owner"], team)}, tok)
        except RuntimeError as e:  # e.g. invitee has not accepted yet -> create without assignee
            print(f"  (assignee rejected, creating unassigned: {str(e)[:120]})")
            created = api("POST", "/issues", payload, tok)
        warn = "" if created["number"] == i["id"] else f"  !! number {created['number']} != planned #{i['id']} (cross-references in texts may be off)"
        print(f"created  #{created['number']} {i['title']}{warn}")


def cmd_assign(tok: str, dry: bool) -> None:
    team = load_team()
    for gh_issue in paginate("/issues?state=open", tok):
        if "pull_request" in gh_issue:
            continue
        owners = [l["name"].split(":", 1)[1] for l in gh_issue["labels"] if l["name"].startswith("owner:")]
        wanted = sorted({u for o in owners for u in assignees_for(o, team)})
        have = {a["login"] for a in gh_issue["assignees"]}
        missing = [u for u in wanted if u not in have]
        if not missing:
            continue
        print(f"assign   #{gh_issue['number']} -> {', '.join(missing)}")
        if not dry:
            try:
                api("POST", f"/issues/{gh_issue['number']}/assignees", {"assignees": missing}, tok)
            except RuntimeError as e:
                print(f"  failed: {str(e)[:160]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["all", "setup", "issues", "assign"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    tok = "" if (a.dry_run and a.command != "assign") else token()
    if a.command in ("all", "setup"):
        cmd_setup(tok, a.dry_run)
    if a.command in ("all", "issues"):
        cmd_issues(tok, a.dry_run)
    if a.command == "assign":
        cmd_assign(tok, a.dry_run)


if __name__ == "__main__":
    main()
