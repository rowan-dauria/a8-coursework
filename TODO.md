# TODO - Tasks before report + repo submission

## A. GitLab repo README

- [x] Create `README.md` at repo root with:
  - [x] One-line project description (GRPO finetune of `google/gemma-3-1b-it` on GSM8K, Tunix/JAX + LoRA).
  - [x] Team members (same names as the report title page: Rowan d'Auria, Basia Koch, Funmi Looi-Somoye).
  - [x] Repo layout: `scripts/` (config, train, rewards, bootstrap, eval), `evaluation/` (per-question jsonls), `docs/RUNS.md`. (Code lives under `tpu-2026/`; README links into it and notes the path mapping.)
  - [x] Setup: venv/deps, how to install (`uv pip ...`), TPU/runtime notes.
  - [x] How to reproduce: train command(s), key env vars (`SEED`, `BETA`, `DATA_SOURCE`, `TPU_CONTENT_DIR`, `WANDB_*`), eval command, bootstrap command. (NB: `NUM_GENERATIONS`/`NUM_BATCHES` are `config.py` constants varied per branch, **not** env vars — README states this accurately.)
  - [x] Branch map: `main` / `n-generations-8` / `reward-reweight` — which run each produced (cross-ref `docs/RUNS.md`).
  - [x] **Clickable** links: W&B project/dashboard, and external datasets/papers (Gemma model, GSM8K paper/TFDS/Kaggle, LoRA paper, Tunix, cmbagent_lg Part II commit).
  - [x] Pointer to where logs live (W&B + jsonls in `evaluation/`).
  - [x] Bonus (marker-focused): full list of report runs (Table 1, K-sweep, other-runs appendices) with clickable W&B links + eval artifacts.

## B. Code to GitLab

- [x] Create the GitLab repo (own namespace).
- [x] Port **all** code from GitHub fork to GitLab (all branches needed for cited runs: `main`, `n-generations-8`, `reward-reweight`).
- [x] Commit **all changes** — no stale mirror; working tree clean.
- [x] Ensure logs are included or reachable: `evaluation/*.jsonl`, `docs/RUNS.md`, W&B links.
- [x] Verify a fresh clone has everything the report references (paths cited: `evaluation/`, `scripts/config.py`, etc.).
- [x] Copy the final GitLab repo URL into the report title page.

## C. Report ready for submission

### Title page / front matter
- [x] Replace `TODO-GitLab-repository` with the real **clickable** GitLab `\href`.
- [x] Replace bare `jgs4c6kl` with a **clickable** W&B dashboard `\href` (project page).
- [x] Team members listed (jointly-owned I.1–I.3 vs individual I.4/Part II clear).
- [x] Resolve the AI-assistance acknowledgement TODO (check local submission policy).

### Links audit (spec: every URL must be a clickable hyperlink)
- [x] All URLs use `\href{}{}` / `\url{}` — GitLab, W&B, datasets, papers. (grep confirms zero bare URLs; all W&B run links, GitLab `\repo`, W&B `\wandb` are wrapped in `\href`. No dataset/paper URLs appear in the source — no references section.)
- [x] Part II: clickable link to the **exact `cmbagent_lg` commit** on the adaptive-planning branch. (report.tex:101-102 — two `\href`s: one to the tree, one to the exact commit `d7d0592`)

### Formatting / page limits (excl. references + appendix)
- [x] A4, 11 pt body, single column, margins ≥ 2 cm, single line spacing. (report.tex: `11pt,a4paper`, `margin=2cm`, single column, default single spacing)
- [ ] I.1–I.3 ≤ 3 pages. **(still spills — content runs pages 2–5 = 4 pages; I.4 starts page 6. Need to cut ~1 page)**
- [x] Part II ≤ 2 pages (workflow diagram excluded). (runs pages 8–9 = 2 pages)
- [x] I.4: no limit but concise. (pages 6–7)
- [x] Whole report typeset in LaTeX; appendices hold no marking-essential material.

### Content (carried over)

#### I.1

#### I.2
- [ ] Re-write in own words

#### I.3
- [ ] Fold in second-seed K=8+new-reward results once VM run + eval complete (update `tab:controlled-comparison`, diagnostic, Limitations).
- [ ]Double check the eval data in the appendicies against Funmi's results. Ensure nothing is missing
- [ ] reference the chat log appendix to show how the training affects the general output of the model.

### Final
- [ ] Build single combined PDF (Part I + Part II).
- [ ] Click-test every hyperlink in the built PDF.
- [ ] Proofread; confirm British English; no em-dashes (per STYLE.md).
