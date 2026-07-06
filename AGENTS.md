<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- Keep changes focused and reviewable.
- Use Conventional Commit PR titles: `type(scope): summary`. Accepted types:
  `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `perf`, `chore`, `revert`,
  `style`, and `build`.
- PR descriptions must include `Summary` and `Validation`.
- Sign every commit with DCO: `git commit -s`.

## Skills

Agent skills live in `.agents/skills/<name>/SKILL.md`. Frontmatter rules,
enforced by `scripts/validate_skills.py` (pre-commit hook `validate-skills`):

- `name` must equal the skill's directory name and be kebab-case.
- `description` must be present, non-empty, and at most 1024 characters.
- `license: Apache-2.0`.
- A `metadata:` block with `author` and `tags`.

Every skill must be listed here (also enforced):

- `debug-session` — start a structured debugging session with a worklog file.
- `dep-create` — create or update Dynamo Enhancement Proposals as GitHub issues.
- `dep-status` — check DEP status and list DEPs by lifecycle state or area.
- `dep-update` — update DEP lifecycle state: triage, PIC, review, approval.
- `dynamo-clone-hotpath-audit` — audit Rust hot-path `.clone()` calls.
- `dynamo-docs` — add, update, move, or remove Dynamo Fern docs content.
- `dynamo-frontend-benchmark` — benchmark and profile the Dynamo frontend.
- `dynamo-interconnect-check` — validate NIXL/UCX/NCCL interconnect readiness.
- `dynamo-recipe-runner` — select, validate, patch, and deploy Kubernetes recipes.
- `dynamo-router-starter` — start or patch router modes and run smoke checks.
- `dynamo-troubleshoot` — diagnose failed or unhealthy Dynamo deployments.
- `gh-issue-bug` — file a GitHub bug issue from conversation context.
- `graham-code-review` — code review in the style of Graham King.
- `pr-monitor` — check CI status and analyze failures for a Dynamo PR.
