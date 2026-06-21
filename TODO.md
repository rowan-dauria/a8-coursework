# TODO - Tasks before report + repo submission

## A. GitLab repo README

- [ ] Create `README.md` at repo root with:
  - [ ] One-line project description (GRPO finetune of `google/gemma-3-1b-it` on GSM8K, Tunix/JAX + LoRA).
  - [ ] Team members (same names as the report title page).
  - [ ] Repo layout: `scripts/` (config, train, rewards, bootstrap, eval), `evaluation/` (per-question jsonls), `docs/RUNS.md`.
  - [ ] Setup: venv/deps, how to install (`uv pip ...`), TPU/runtime notes.
  - [ ] How to reproduce: train command(s), key env vars (`SEED`, `NUM_GENERATIONS`, `NUM_BATCHES`, `TPU_CONTENT_DIR`, `WANDB_*`), eval command, bootstrap command.
  - [ ] Branch map: `main` / `n-generations-8` / `reward-reweight` — which run each produced (cross-ref `docs/RUNS.md`).
  - [ ] **Clickable** links: W&B project/dashboard, and any external datasets/papers cited in the report.
  - [ ] Pointer to where logs live (W&B + jsonls in `evaluation/`).

## B. Code to GitLab

- [ ] Create the GitLab repo (own namespace).
- [ ] Port **all** code from GitHub fork to GitLab (all branches needed for cited runs: `main`, `n-generations-8`, `reward-reweight`).
- [ ] Commit **all changes** — no stale mirror; working tree clean.
- [ ] Ensure logs are included or reachable: `evaluation/*.jsonl`, `docs/RUNS.md`, W&B links.
- [ ] Verify a fresh clone has everything the report references (paths cited: `evaluation/`, `scripts/config.py`, etc.).
- [x] Copy the final GitLab repo URL into the report title page.

## C. Report ready for submission

### Title page / front matter
- [x] Replace `TODO-GitLab-repository` with the real **clickable** GitLab `\href`.
- [x] Replace bare `jgs4c6kl` with a **clickable** W&B dashboard `\href` (project page).
- [ ] Team members listed (jointly-owned I.1–I.3 vs individual I.4/Part II clear).
- [x] Resolve the AI-assistance acknowledgement TODO (check local submission policy).

### Links audit (spec: every URL must be a clickable hyperlink)
- [ ] All URLs use `\href{}{}` / `\url{}` — GitLab, W&B, datasets, papers.
- [ ] Part II: clickable link to the **exact `cmbagent_lg` commit** on the adaptive-planning branch.

### Formatting / page limits (excl. references + appendix)
- [ ] A4, 11 pt body, single column, margins ≥ 2 cm, single line spacing.
- [ ] I.1–I.3 ≤ 3 pages. **(currently spills onto page 4 — fix)**
- [ ] Part II ≤ 2 pages (workflow diagram excluded).
- [ ] I.4: no limit but concise.
- [ ] Whole report typeset in LaTeX; appendices hold no marking-essential material.

### Content (carried over)

#### I.1

#### I.3
- [ ] Fold in second-seed K=8+new-reward results once VM run + eval complete (update `tab:controlled-comparison`, diagnostic, Limitations).

### Final
- [ ] Build single combined PDF (Part I + Part II).
- [ ] Click-test every hyperlink in the built PDF.
- [ ] Proofread; confirm British English; no em-dashes (per STYLE.md).
