# Report-Level Requirements

Source: `notes/a8-coursework.md`. This file isolates requirements that apply to the submitted report as a whole or to broad report parts. It intentionally excludes section-specific question prompts and marking rubrics.

## Submission Artifact

- Submit a single PDF report covering both Part I and Part II.
- The report should be written in LaTeX so the submission is one coherent document.
- Section I.4 must be typeset in LaTeX. No scanned handwriting, photographed whiteboards, or Word equations.
- Use `amsmath` environments such as `align` and `equation` for mathematics, and number any equation that is referenced later.

## Page Limits and Formatting

- Part I practical write-up, sections I.1-I.3: at most 3 pages.
- Part I theory, section I.4: no page limit, but answers must be clear and concise.
- Part II: at most 2 pages, excluding the workflow diagram.
- Page limits exclude references and appendix.
- Unless stated otherwise, use A4 paper, 11 pt body font, single column, margins of at least 2 cm, and single line spacing.
- Do not shrink the font, margins, or line spacing to evade page limits.
- Appendices do not count toward page limits, but must not contain material essential to marking.

## Individual Authorship and Team Context

- Every student submits their own individual PDF.
- Sections I.1-I.3 describe a team project, but the practical write-up must be written in the student's own words.
- Team members may legitimately emphasise different runs, plots, or interpretations of shared experiments.
- Do not submit the same prose as a teammate.
- State team members explicitly on the title page so the marker can identify jointly owned experiments.
- Section I.4 and Part II are strictly individual work.

## Links and External Material

- Every URL in the report must be a clickable hyperlink in the submitted PDF.
- Use `\href{...}{...}` or `\url{...}` for report links.
- Clickable URLs are required for the GitLab repository, W&B runs, dashboards, dataset cards, papers, external code, baselines, and any other external cited material.
- Anything cited outside the report must have a clickable URL the marker can follow without retyping.

## Code and Logs

- Part I code may be developed on GitHub, but the final code submission must be ported to the student's GitLab repository.
- The GitLab repository constitutes the submitted code repository.
- Link the GitLab repository from the report.
- All training and evaluation logs, such as W&B, TensorBoard, JSONL, or equivalent records, must be included in or made accessible from the GitLab repository.
- For the Part I practical deliverable, include a clickable link to the team's GitLab repository with all changes committed.

## AI Assistance and Accountability

- AI assistance while writing code is expected and permitted.
- The student remains fully responsible for every submitted line of code and must be able to explain it in a viva.

## Marking Context

- The exam has 120 total marks.
- Part I practical, sections I.1-I.3, is marked out of 50.
- Part I theory, section I.4, is marked out of 40.
- Part II is marked out of 30.
- The final mark is the sum of these components divided by 120.
