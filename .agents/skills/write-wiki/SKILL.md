---
name: write-wiki
description: Generates a directory-scoped WIKI.md capturing architecture, ontology, invariants, debugging history, and operational shortcuts. Use when user says "write a wiki", "create wiki", "document what we learned", or invokes /write-wiki.
---

# Write Wiki

Generate a `WIKI.md` file for a target directory by mining conversation history, touched files, and discovered insights.

## Quick Start

1. Identify the **target directory** — the folder the wiki will live in.
2. Mine sources (see Mining Order below).
3. Generate `WIKI.md` using the flexible section menu from [REFERENCE.md](REFERENCE.md).
4. Place the file at `<target_directory>/WIKI.md`.

## Mining Order

Gather material from these sources, in priority order:

### Source 1: Current Conversation
Extract insights, fixes, ontological discoveries, and invariants surfaced in the current session.

### Source 2: Conversation History Mining (CRITICAL — do not skip)

Conversations reveal pain points, stuck moments, debugging narratives, and the rules established when issues were resolved. Mine them systematically:

Transcripts come from one of two places. Use the one that matches the tool you are running in. If both exist, search both.

- **T3 Code** (any provider: Claude, Codex, Command Code and other ACP agents, Antigravity run inside T3). T3 stores every thread in one SQLite database, `~/.t3/userdata/statev2.sqlite`. Use only this; do not also search the providers' own transcript folders (`~/.claude*/projects`, etc.), which hold copies of the same sessions.
- **Antigravity (standalone)**: `~/.gemini/antigravity*/brain/<conv-id>/.system_generated/logs/transcript.jsonl`.

**Step 1 — Find related conversations.** Search for each key file in the target directory (e.g. `bookings.service.js`, `availability.service.js`).

T3 — tool calls, file reads and edits live in `orchestration_v2_projection_turn_items`. Always open the database read-only, since T3 is writing to it:
```bash
sqlite3 -readonly "file:$HOME/.t3/userdata/statev2.sqlite?mode=ro" "
SELECT t.thread_id, t.default_provider, substr(t.updated_at,1,10), t.title
FROM orchestration_v2_projection_threads t
JOIN projection_projects p ON p.project_id = t.project_id
WHERE p.workspace_root = '<absolute repo root>'
  AND t.deleted_at IS NULL
  AND t.thread_id IN (
    SELECT COALESCE(json_extract(c.payload_json,'$.lineage.parentThreadId'), c.thread_id)
    FROM orchestration_v2_projection_threads c
    WHERE c.thread_id IN (SELECT thread_id FROM orchestration_v2_projection_turn_items
                          WHERE payload_json LIKE '%<filename>%'))
ORDER BY t.updated_at DESC LIMIT 20;"
```
A match inside a subagent thread is reported as its parent thread, so every result is a top-level thread you can cite and open.
Do not rely on `t3_thread_search` for this step: it matches titles and message text only, not file names inside tool calls. The thread you are running in will match too; skip it.

Antigravity:
```bash
grep -rl "<filename>" ~/.gemini/antigravity*/brain/*/.system_generated/logs/transcript.jsonl 2>/dev/null
```
The `<conv-id>` is the folder name under `brain/`.

**Step 2 — Triage by first user message.** Assess each conversation's opening request and skip ones that are trivially unrelated (formatting tweaks, unrelated features). The T3 title from Step 1 is often enough; otherwise:

T3:
```bash
sqlite3 -readonly "file:$HOME/.t3/userdata/statev2.sqlite?mode=ro" "
SELECT substr(json_extract(payload_json,'$.text'),1,300)
FROM orchestration_v2_projection_messages
WHERE thread_id = '<thread_id>' AND role = 'user'
ORDER BY created_at LIMIT 1;"
```
Antigravity:
```bash
grep '"type":"USER_INPUT"' ~/.gemini/antigravity*/brain/<conv-id>/.system_generated/logs/transcript.jsonl | head -1
```

**Step 3 — Also check injected conversation summaries.** The system injects recent conversation summaries at conversation start. Review them for conversations that mention the target directory's domain even if they touched different files.

**Step 4 — Read relevant transcripts.** For the triaged conversations (aim for the top 5–10 most relevant), read them. In T3, use `t3_thread_read` with the `threadId` (`view: "messages"` for the conversation, `view: "activity"` for tool calls and edits; keep `limit` and `maxCharsPerItem` low and page with `afterPosition`). In Antigravity, read the `transcript.jsonl`. Extract:
- **The symptom / stuck point** — what went wrong or what the user was trying to achieve.
- **The reasoning trace** — what was investigated, what hypotheses were tested.
- **The resolution** — what code change or rule was established.
- **New invariants or constraints** — any rules the user explicitly stated or that emerged from the fix.
- **The source** — the conversation id and, if the fix was committed, the short commit hash (`git log --oneline -- <file>` around the conversation's date). Every entry you write from this conversation carries it (see Writing Rule 10).

**Step 5 — Cross-reference with current code.** Verify that insights from past conversations still hold against the current codebase. Past resolutions may have been superseded.

### Source 3: Files in Scope
Read key files in the target directory and any cross-stack files touched during sessions (backends, services, scripts, configs).

### Source 4: Knowledge Items
Check KIs for overlapping domain knowledge that should be referenced (but not duplicated).


## Section Menu (Flexible)

Pick the sections that apply. Not every directory needs all sections. Adapt section names to the domain (e.g., "State Management Invariants" instead of "Ontology" for a UI component folder).

| Section | Include When |
|---------|-------------|
| Domain Architecture & File Structure | Always — every wiki needs a file map |
| Fast-Path Diagnostics & Shortcuts | Connection strings, cache gotchas, CLI commands were discovered |
| Domain Ontologies, Mechanics & Behavioral Relationships | Domain terms, evaluation hierarchies, priority stacks, or operational modes exist |
| Entity/Data Relationships & Rules | Mapping logic, lookup tables, type discriminators exist |
| Historical Issues & Reasoning Traces | Bugs were debugged with non-obvious root causes |
| Diagnostics & Troubleshooting Utilities | Scratch scripts or debug commands were used |

## Writing Rules

1. **Domain Ontologies & Ontological Precision** — define domain ontologies with formal boundaries, operational differences, type discriminators, and entity definitions.
2. **Architecture, Hierarchy & Priority Stacks** — document evaluation hierarchies, priority stacks (e.g. Date Overrides > Slot Overrides > Weekly Schedule > Defaults), fallback chains, and routing mechanics.
3. **Mechanics, Behavioral Relationships & Specific Constraints** — document mode-specific mechanics, behavior relationships, and specific constraints across different operational contexts (e.g., single physical court 1:1 reservations vs session attachments vs multi-court pooled auto-balancing).
4. **Invariant Capture** — call out boolean rules and strict system guarantees that must always hold, separately from definitions and descriptions.
5. **Reasoning Traces** — document bugs as **Symptom → Reasoning Trace → Resolution**, not just "what changed."
6. **Cross-stack linking** — link files across backend, frontend, and scripts using absolute `file://` paths.
7. **Use mermaid diagrams** for architecture/data flow when there are 3+ files in the pipeline.
8. **Use tables** for mappings, column definitions, priority matrices, and behavioral relationships.
9. **Use `> [!IMPORTANT]`** alerts for gotchas and cache invalidation traps.
10. **Source links** — every Historical Issue, invariant and priority rule names the evidence it came from, so a later agent can check it instead of trusting it. Format: `t3:<thread_id>` or `antigravity:<conv-id>`, plus ` · <short commit>` when there is one; use `code` when the claim was read straight from the code and no conversation produced it. Put it on a `* **Source**:` line for Historical Issues, and as a trailing `*(Source: …)*` on invariants and rules. Never invent a source; if you cannot find one, write `Source: unknown`.

## Wiki vs Knowledge Items

Do **not** duplicate KI content. The boundary:
- **KI** = Cross-project or cross-directory insight (e.g., "Vite mkcert SSL mismatch").
- **WIKI.md** = Directory-scoped operational knowledge (e.g., "how the reports pipeline works and its invariants").

If a KI covers the same ground, reference it rather than restating it.

## Output

Place the generated file at `<target_directory>/WIKI.md`. Always present the result for user review.

Add this note directly under the wiki's header, so readers of the wiki know how to use the sources:

```markdown
> Entries cite their evidence as `t3:<thread_id>` (open with `t3_thread_read`), `antigravity:<conv-id>` (`~/.gemini/antigravity*/brain/<conv-id>/`), a commit hash, or `code`. If an entry is vague, conflicts with another entry, or does not match what you see, open its source before trusting either.
```

See [REFERENCE.md](REFERENCE.md) for the full section template and the examples directory for concrete exemplars:
- [reports_wiki.md](examples/reports_wiki.md) — financial ontology, column mapping matrices, entity relationship rules.
- [services_wiki.md](examples/services_wiki.md) — domain ontologies with formal boundaries, priority resolution hierarchies, behavioral relationship tables, system invariants.
