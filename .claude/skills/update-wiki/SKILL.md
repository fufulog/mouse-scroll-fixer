---
name: update-wiki
description: Lightweight update of an existing directory-scoped WIKI.md with new insights from the current session. Use when a debugging session completes, a fix is confirmed, or the agent has used a WIKI.md during work and discovered something new. Does NOT create wikis — only updates existing ones.
---

# Update Wiki

Lightweight companion to `write-wiki`. Adds new insights to an **existing** `WIKI.md` without re-mining full conversation history.

## Preconditions (ALL must be true)

1. A `WIKI.md` **already exists** in the target directory. If none exists, **do nothing** — do not create one.
2. The agent **has used the wiki** during this session (read it, referenced it, or operated in the directory it covers).
3. The agent **has something useful to add** — a new insight, invariant, bug fix, or correction discovered during the current session.

If any precondition fails, skip silently. Do not prompt the user.

## What Qualifies as "Useful"

Add to the wiki when any of these occurred in the current session:

- A **domain term or formal boundary** was clarified → add to Domain Ontologies
- An **evaluation hierarchy or priority stack** was uncovered → add to Priority & Resolution Hierarchy
- A **mode-specific mechanic, behavior relationship, or specific constraint** was added or modified → add to Mechanics & Behavioral Relationships
- A **strict boolean guarantee or system rule** was discovered or enforced → add to Key System Invariants
- A **bug was fixed** with a non-obvious root cause → add to Historical Issues
- A **new file became relevant** to the pipeline → add to Domain Architecture & File Structure
- A **cache gotcha or diagnostic shortcut** was discovered → add to Fast-Path Local Diagnostics
- A **scratch script was created** for debugging → add to Diagnostics Utilities
- A **structural rule or lookup mapping** was added or changed → add to Entity Relationships
- An **existing section became stale** due to code changes → update the section
- An entry's **source contradicted it** when checked → correct the entry and cite the evidence that settled it

## Update Strategy

### Adding New Content

- **Historical Issues**: Append new entries. Never remove old ones unless the subsystem is deleted.
- **Architecture & Files**: Add new files, update mermaid diagrams if the data flow changed.
- **Domain Ontologies & Mechanics**: Add new terms, update formal boundaries, extend priority hierarchies, add behavioral relationships, or refine specific constraints.
- **New sections**: If the insight doesn't fit existing sections, create a new section following the flexible menu from `write-wiki`.

### Source Links

Every entry you add or change carries its source, in the format defined in `write-wiki` (Writing Rule 10 and REFERENCE.md "Source Format"):

- From this session in T3: `t3:<current thread_id>` (the id of the thread you are running in), plus ` · <short commit>` once the fix is committed.
- From this session in standalone Antigravity: `antigravity:<conv-id>`.
- Read straight from the code: `code`.
- Historical Issues get a `* **Source**:` line; invariants and rules get a trailing `*(Source: …)*`.
- When you correct an entry, keep its old source and add the new one: `*(Source: t3:aaa · 1234abc; corrected t3:bbb · 5678def)*`.
- If the wiki has no source note under its header, add the one from `write-wiki`'s Output section. Do not backfill sources for old entries you did not touch.

### Checking Before Trusting

When an entry is vague, conflicts with another entry, or does not match what you see in the code or the session, open its source before acting on it (`t3_thread_read` for `t3:` ids, the `brain/<conv-id>/` transcript for `antigravity:` ids, `git show` for commits). Then either correct the entry or note why the observed behavior differs. Entries marked `unknown` get the least trust.

### Updating Existing Content

- If code changes make an existing section **factually wrong**, update it to reflect current state.
- If a section describes something that **no longer exists** (e.g., a removed CSV exporter), move the knowledge to a `### Archived` subsection within that section — do not delete it.
- Preserve all existing content that is still accurate.

### What NOT to Do

- Do **not** rewrite sections that are still correct just for style.
- Do **not** create a WIKI.md if none exists.
- Do **not** add trivial changes (typo fixes, formatting tweaks) as Historical Issues.
- Do **not** duplicate Knowledge Item (KI) content — reference KIs instead.

## Process

1. Read the existing `WIKI.md`.
2. Identify which sections are affected by the current session's insights. For any entry the session contradicts, open its source first (see Checking Before Trusting).
3. Apply targeted edits — append new entries or update stale content, each with its source.
4. Present the changes for user review.
