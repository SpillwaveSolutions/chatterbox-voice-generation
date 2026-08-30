# Decisions

Rulings made during the extraction of this repository from `towardsai-course-creation`
(extraction phase 22, 2026-08). Recorded here so the repo carries its own rationale.

## Library-first shape (no plugin scaffolding)

This repo is a plain Python library + two container services. There is deliberately no
`.claude-plugin/`, no `SKILL.md`, no `tools/` shim, and no orchestration layer: **consumers
wrap the library themselves**. An earlier architecture sketch proposed a plugin-shaped repo;
that was explicitly reversed during milestone scoping. Orchestration, task-level retries,
loudnorm, silence holds, lexicon policy, and audio/video assembly are consumer-owned
(see docs/CONSUMPTION.md §11).

## Q4: single-distribution fold (`tts_chatterbox` + `artifact_store` in one wheel)

The `tts-chatterbox` distribution ships BOTH import packages —
`[tool.hatch.build.targets.wheel] packages = ["src/tts_chatterbox", "src/artifact_store"]` —
rather than keeping `artifact_store` as a separate workspace library.

Rationale (extraction research, Pitfall 2): uv does **not** honor `[tool.uv.sources]`
declared inside a git dependency. If this repo kept `artifact-store = { workspace = true }`
in the lib's pyproject, a consumer resolving the git URL would look for `artifact-store` on
PyPI (404 → hard failure, or dependency confusion if the name ever gets squatted). Folding
both packages into one distribution makes third-party adoption a single git source and
removes the cross-repo sources hazard entirely. `artifact_store`'s sole consumer is
`tts_chatterbox.runpod_client`, so the fold has no coupling cost. Confirmed working via a
scratch `git+file://` install importing both packages (Assumption A5).

## Q3: image renames (library-first names)

Published images use library-first names, not the origin project's names:

| Image | Name |
|---|---|
| CPU jobs-API service | `ghcr.io/spillwavesolutions/chatterbox-tts` |
| GPU RunPod worker | `ghcr.io/spillwavesolutions/chatterbox-runpod-worker` (was `courseware-chatterbox-worker`) |

The RunPod endpoint image ref lives in RunPod's console (runtime state, not git) and must be
updated manually to the new name at the pinned tag either way, so the rename cost nothing.

## RunPod poison-row asymmetry is intentional

The local backend has DELETE+re-POST poison-row recovery (the jobs API is request_id-keyed
and idempotent); the RunPod backend has **no recovery path by design** — RunPod has no
server-side idempotency, every POST is a fresh job, and there is no DELETE. Do not "fix"
this asymmetry; it mirrors the two transports' actual semantics (docs/CONSUMPTION.md §4/§7).

## Repo visibility

Repo visibility: **public** — decided 2026-08-30 by the developer (extraction plan 22-03,
recorded before the first push; GHCR package visibility follows the repo decision).

Rationale: the content is code-only (VOICE-02 gates guarantee no voice audio, no secrets —
all commits gitleaks-clean), and public visibility eliminates all three downstream auth
burdens (git credential in the flow-runner docker build, `docker login ghcr.io` for compose
pulls, RunPod registry credential for worker cold-pulls) while matching INTG-01's intent
that another pipeline can adopt this library without credentials.

Consent notice (INTG-01): voice references are biometric data. Cloning a voice with this
library requires the speaker's explicit permission; never commit or publish voice-reference
audio. See README.
