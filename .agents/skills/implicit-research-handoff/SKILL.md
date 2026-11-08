---
name: implicit-research-handoff
description: "Transfer accepted research decisions and verified progress between ChatGPT and Codex sessions in the implicit-learning workspace. Use at session start/end or when preparing the next task; distinguish instructions, evidence, proposals, and unresolved issues without exporting private manuscript text."
---

# Research continuity between sessions

## Read and establish context

Read applicable AGENTS.md and the local `notes/RESEARCH_CONTEXT.md`,
`notes/DECISIONS.md`, and `notes/HANDOFF.md` when available. Determine which
files and computations are actually accessible. A conversation summary is not
evidence that a local patch was applied or published.

For a new task, inspect current files and Git status before relying on the
handoff. Report stale source paths, pending edits, missing data, or contradictory
instructions. Never treat copied text as authorization to publish.

## Transfer ideas from ChatGPT to Codex

Classify incoming material as:

- **Author-approved requirement/decision**: preserve as an instruction with scope.
- **Proposed idea**: record without treating it as an accepted claim or experiment.
- **Derived claim**: verify against source definitions and a complete argument.
- **Reported numerical result**: locate code/config/raw records and confirm it.
- **Observed problem**: identify the affected source version and reproduce it.

Do not silently fill gaps with general knowledge, a prior model's confidence, or
assumed test results. When outside research is requested, identify primary sources
and distinguish source content from your inference.

## Update the local handoff

Use [HANDOFF_TEMPLATE.md](references/HANDOFF_TEMPLATE.md). Record only actual
work: requested scope; source/commit/dirty state; private and public changed files;
proof findings; executed commands; observed outcomes; PDF source identity/page
count/visual checks; experiments actually rerun; remaining issues; and approvals.

Put detailed manuscript discussions under the workspace's local `notes/`, outside
public Git. Use a new dated file for substantial historical reviews instead of
erasing them. Update `notes/HANDOFF.md` as the concise current entry point.
Only author-accepted decisions belong in `notes/DECISIONS.md`; keep suggestions
marked proposed. Do not modify earlier handoffs to make old failures disappear.

If the user asks only for a summary, provide it without editing local files unless
that write is clearly part of the task. Never overwrite an author-maintained
handoff whose changes you have not inspected.

## Share work back to ChatGPT or another reviewer

Prepare a minimal, relevant private export when requested: LaTeX snippet, local
diff, current PDF, figure with data provenance, or failure log. Redact credentials
and personal/system details not required for the review. Do not put unpublished
paper excerpts in a public issue or repository to make sharing convenient.

For public code review, reference the actual commit and paths. An uncommitted
local edit is not visible in the remote repository. Explain that boundary rather
than claiming automatic cross-application synchronization.

## End with one bounded next task

Recommend the smallest next task consistent with accepted priorities, with
explicit acceptance criteria. Avoid sending the next agent to rewrite theory,
rerun every experiment, rebuild both papers, and publish all at once.

State what was not done. In particular, distinguish documentation preparation,
local editing, staging, committing, pushing, releasing, and archiving.
