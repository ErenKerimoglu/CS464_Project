---
id: 1
title: "[Setup] Team onboarding & environment setup (everyone)"
owner: all
labels: [type:setup, priority:critical]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
Every member can clone the repo, install the environment, run the tests, and understands the workflow. **Due: first 2 days.**

## Tasks (each member ticks their own line)
- [ ] Eren — accepted invite · env installed · `pytest` passes · read docs
- [ ] Ege — accepted invite · env installed · `pytest` passes · read docs
- [ ] Yunus — accepted invite · env installed · `pytest` passes · read docs
- [ ] Berkay — accepted invite · env installed · `pytest` passes · read docs
- [ ] Eftelya — accepted invite · env installed · `pytest` passes · read docs

Steps for each member:
1. Accept the GitHub collaborator invitation.
2. `git clone`, then `python -m venv .venv`, then `pip install -e ".[dev]"` (add `".[all]"` on a GPU machine).
3. `pytest -q` and `ruff check .` both pass.
4. Read `docs/PROJECT_PLAN.md` (especially §3 protocol and §4 contracts), `docs/WORK_SEPARATION.md`, `CONTRIBUTING.md`.
5. Create a Kaggle account with phone verification (needed for GPU) and join the team's private Kaggle Dataset once #10 shares it.
6. Comment your GitHub username here so Eren can run `python tools/github/create_issues.py assign`.

## Acceptance criteria
- All five lines are ticked, and every member is assigned to their issues on GitHub.
