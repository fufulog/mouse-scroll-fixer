# WIKI.md Reference Template

This document provides the full section template for generating `WIKI.md` files. Each section includes field guidance, formatting rules, and extraction heuristics.

---

## Document Header

```markdown
# <Directory Purpose> Wiki: Architecture, Invariants, & Troubleshooting

This wiki documents the structural design, data flow patterns, system invariants, and debugging history of <scope description>.

> Entries cite their evidence as `t3:<thread_id>` (open with `t3_thread_read`), `antigravity:<conv-id>` (`~/.gemini/antigravity*/brain/<conv-id>/`), a commit hash, or `code`. If an entry is vague, conflicts with another entry, or does not match what you see, open its source before trusting either.
```

### Source Format (used throughout)

| Evidence | Format | How to open it |
| :--- | :--- | :--- |
| T3 thread (any provider) | `t3:<thread_id>` | `t3_thread_read` with `threadId` |
| Antigravity conversation | `antigravity:<conv-id>` | `~/.gemini/antigravity*/brain/<conv-id>/.system_generated/logs/transcript.jsonl` |
| Commit | `· <short hash>` appended to either of the above, or alone | `git show <hash>` |
| Read from the code directly | `code` | the linked file |
| Not found | `unknown` | — |

---

## Section 1: Domain Architecture & File Structure

**Purpose**: Map the data flow pipeline and name every key file with clickable links.

**Always include this section.**

### Template

```markdown
## 1. Domain Architecture & File Structure

<One-sentence description of what this directory/system bridges.>

\`\`\`mermaid
graph TD
    <nodes and edges showing data flow>
\`\`\`

### Key Files and Directories

* **<Layer/Category>**:
  * [FileName.ext](relative/path/from/the/wiki): <One-line purpose description.>
```

### Extraction Rules

- Include files from **all layers** touched during sessions (backend, frontend, scripts, configs).
- Use mermaid `graph TD` for pipelines with 3+ files.
- Group files by layer or category (Backend Repository, Backend DTOs, Frontend Components, etc.).
- Every file link must be an absolute `file://` path.

---

## Section 2: Fast-Path Local Diagnostics & Shortcuts

**Purpose**: Eliminate wasted time in future sessions by documenting connection strings, cache behaviors, and CLI shortcuts.

**Include when**: Environment-specific gotchas, connection details, or cache invalidation traps were discovered.

### Template

```markdown
## 2. Fast-Path Local Diagnostics & Shortcuts

* **<Resource Name>**: 
  `<connection string or CLI command>`
* **<Gotcha Name>**:
  > [!IMPORTANT]
  > <Description of the trap and how to avoid it.>
```

### Extraction Rules

- Pull connection strings, port numbers, and credentials from `.env`, config files, or conversation context.
- Document any cache TTLs and how to invalidate them.
- Include CLI one-liners that were repeatedly used during debugging.

---

## Section 3: Domain Ontologies, Mechanics & Behavioral Relationships

**Purpose**: Document formal boundaries, domain ontologies, operational differences, evaluation hierarchies, and mode-specific mechanics across the domain.

**Include when**: Domain entities, priority stacks, evaluation hierarchies, or operational modes exist.

### Template

```markdown
## 3. <Domain> Domain Ontologies & Operational Mechanics

<One-sentence description of formal boundaries and domain concepts.>

### Domain Ontologies & Discriminators

| Term / Discriminator | Formal Boundary & Definition | Operational Differences | Key Properties |
| :--- | :--- | :--- | :--- |
| **`<term>`** | <Formal definition and domain scope.> | <How behavior or execution differs at runtime.> | <Specific fields, flags, or configuration.> |

### Priority & Resolution Hierarchy (Highest to Lowest)

1. **<Level 1 — Override>**: <Description of the highest-priority rule or override.> *(Source: <source>)*
2. **<Level 2 — Configured>**: <Fallback or secondary resolution rule.>
3. **<Level N — Base Default>**: <Default behavior when no higher rule applies.>

### Behavioral Relationships, Mechanics & Specific Constraints

| Operational Context | Mechanics | Specific Constraints | Authority / Override Behavior |
| :--- | :--- | :--- | :--- |
| **<Context>** | <How resources are allocated, routed, or composed.> | <Capacity limits, validation rules, or formulas.> | <Who can bypass, what gets enforced regardless.> |

### Key System Invariants

* **<Invariant Name>**: <Strict boolean guarantee that must always hold, stated precisely.> *(Source: <source>)*
```

### Extraction Rules

- Document formal boundaries and operational differences between domain terms (Domain Ontologies).
- Define evaluation hierarchies, priority stacks, and fallback chains explicitly.
- Separate **definitions** (what things are) from **mechanics & behavioral relationships** (how they execute) and **invariants** (what must always hold).
- Adapt column headers to the domain — e.g. "Allocation Mechanics" for resource systems, "State Transition Rules" for workflows, "Pricing Logic" for financial pipelines.
- Name each invariant for easy reference.
- End each invariant and hierarchy level with its source (see Source Format).

---

## Section 4: Entity/Data Relationships & Rules

**Purpose**: Document how entity types map to pricing, IDs, or other lookup structures.

**Include when**: Mapping logic, lookup tables, or type discriminators exist in the codebase.

### Template

```markdown
## 4. Entity Relationships & <Domain> Rules

<Context sentence about the mapping system.>

### <Mapping Name> Rules

| `discriminator` | Target Column | Key Pattern | Source Field |
| :--- | :--- | :--- | :--- |

### Key Structural Rules

1. **<Rule Name>**: <Precise description of the rule and when it applies.>
```

### Extraction Rules

- Build lookup tables from code patterns (switch statements, if-else chains, enum mappings).
- Number structural rules for easy referencing.
- Call out edge cases and exclusions explicitly.

---

## Section 5: Historical Issues & Reasoning Traces

**Purpose**: Document past bugs as full reasoning chains so the same class of error is never re-investigated from scratch.

**Include when**: Bugs were debugged with non-obvious root causes.

### Template

```markdown
## 5. Historical Issues & Reasoning Traces

### <Issue Title>
* **Symptom**: <What the user observed or what the data showed.>
* **Reasoning Trace**: 
  1. <First hypothesis or investigation step.>
  2. <What was discovered.>
  3. <Root cause identification.>
* **Resolution**: <What was changed and why.>
* **Source**: t3:<thread_id> · <short commit>
```

### Extraction Rules

- Never reduce a bug to just "fixed X" — always include the **reasoning chain**.
- Include the symptom as the user experienced it, not as a technical description.
- Number the reasoning steps chronologically.
- Always include the **Source** line (see Source Format). The commit goes there, not in the prose.

### Lifecycle Rules

- **New issues are appended** — never remove historical entries unless the entire subsystem they describe is deleted.
- **Superseded issues** should be annotated, not removed: `*(Superseded by issue below)*`.
- If the underlying system changes such that an issue is no longer possible, move it to a `### Archived Issues` subsection rather than deleting it.

---

## Section 6: Diagnostics & Troubleshooting Utilities

**Purpose**: Point to scratch scripts and debug tools used during sessions.

**Include when**: Diagnostic scripts or debug commands were created or used.

### Template

```markdown
## 6. Diagnostics & Troubleshooting Utilities

The `/<directory>` contains diagnostic scripts used to debug <domain> issues:

* **[script_name.py](relative/path/from/the/wiki)**: <One-line description of what it does.>
```

### Extraction Rules

- Link to scripts with absolute `file://` paths.
- Describe each script's purpose in one line — focus on **what question it answers**, not implementation details.
- Include both permanent utilities and scratch scripts.

---

## Flexible Sections

The six sections above are the **menu**, not a rigid template. Adapt freely:

| Standard Section | Alternative Names for Other Domains |
|-----------------|-------------------------------------|
| Domain Ontologies, Mechanics & Behavioral Relationships | State Machine Rules, Data Model Semantics, API Contract Definitions, Scheduling Architecture |
| Entity Relationships | Component Hierarchy, Route Mapping, Store Dependencies, Pricing Rules |
| Fast-Path Diagnostics | Development Environment Shortcuts, Local Testing Recipes |

Create new section types when the domain calls for it. The structural rules (precision, hierarchies, mechanics, invariants, reasoning traces) apply regardless of section name.
