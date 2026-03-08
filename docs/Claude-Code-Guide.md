# Claude Code Developer Guide

## Introduction

[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) is Anthropic's CLI tool for AI-assisted software development. It reads your code, runs terminal commands, and edits files based on natural-language requests.

This repository includes a configuration layer (`.claude/` directory and `CLAUDE.md`) that teaches Claude Code about cat-detector's conventions: how we build, our Rust style, commit standards, deploy workflow, and so on. Without this config, Claude still works — it just won't know project-specific details and will make more mistakes.

Nothing here is mandatory. If you don't use Claude Code, these files don't affect your workflow. If you do, the config is opt-in infrastructure that reduces the amount of context you need to repeat in every conversation.

> **Note:** Best practices for configuring Claude Code are still evolving as the tool matures. This guide reflects what works well today; expect the config to adapt as Anthropic ships new features and the community discovers better patterns.
> For official Claude Code documentation, see: <https://docs.anthropic.com/en/docs/claude-code/overview>

---

## How the Configuration Is Organized

The configuration is split across several files so Claude can load only what's relevant to a given task, rather than consuming its entire context window on instructions it doesn't need.

| Artifact | Location | Purpose |
|----------|----------|---------|
| Root instructions | `CLAUDE.md` | Top-level entry point — kept short, links to everything else |
| Rules | `.claude/rules/` | Guidance Claude reads when working on code (Rust style, testing, git workflow) |
| Settings | `.claude/settings.json` | Permissions — what Claude can run without asking |
| Skills | `.claude/skills/` | Slash-command definitions for common workflows (`/build`, `/test`, `/deploy`, etc.) |
| Agents | `.claude/agents/` | Subagent definitions — specialized roles Claude can delegate to |

**Why is CLAUDE.md so short?** The root `CLAUDE.md` has global scope — Claude reads it on every interaction regardless of what you're working on. Every token there consumes context window space that could hold your actual code. So `CLAUDE.md` is kept to a minimal routing table that points to modular rule files, which Claude loads selectively based on which files you're editing. This follows [community best practices for Claude Code configuration](https://github.com/shanraisshan/claude-code-best-practice).

---

## Rules: Teaching Claude the Project Conventions

Rules live in `.claude/rules/` as plain Markdown files. Claude reads them when working on related code and follows the guidance they contain.

| Rule file | What it covers |
|-----------|---------------|
| `git-workflow.md` | Commit messages (imperative mood, Co-Authored-By), pre-commit checks, what never to commit |
| `rust-style.md` | Error handling (`thiserror`/`anyhow`), trait-based DI, async patterns, feature gates |
| `testing.md` | Test placement, naming convention, Mock* implementations, what to test and not test |

**Rules are suggestions, not enforcement.** Claude reads `.claude/rules/git-workflow.md` and knows it should run `cargo fmt --check` before committing. But under context pressure — long conversations, complex multi-step tasks — the model can deprioritize rules it considers optional. A rule says "you should"; it doesn't guarantee "you will."

If you need mechanical enforcement (e.g., blocking commits that fail linting), you can add hooks — scripts that intercept tool calls and block them on failure. This project doesn't currently use hooks, but they can be added to `.claude/settings.json` under the `hooks.PreToolUse` key.

**Rules can drift,** but they're easy to maintain:

- Update rules in the same PR that introduces a new pattern.
- Rules are plain Markdown — they're reviewed in PRs just like code.
- You can ask Claude to audit rules against recent changes: "Read the rules in `.claude/rules/` and check if any are outdated."

---

## Customizing Claude for Your Workflow

Claude Code settings exist at four layers, from broadest to narrowest:

1. **Project settings** (`.claude/settings.json`) — checked into the repo, shared by the team. Defines default permissions.
2. **Personal global settings** (`~/.claude/settings.json`) — your machine-wide preferences across all projects.
3. **Personal project settings** (`~/.claude/projects/<project>/settings.json`) — your per-project overrides stored outside the repo.
4. **Personal local settings** (`.claude/settings.local.json`) — per-project overrides, gitignored. Your customizations that don't affect teammates.

For example, the project settings pre-approve read-only commands like `cargo build`, `cargo test`, `git status`, and `ssh catbox`. If you want to also pre-approve Python scripts or other tools, add them to your personal local settings (`.claude/settings.local.json`):

```json
{
  "permissions": {
    "allow": [
      "Bash(python3 scripts/*)",
      "WebSearch"
    ]
  }
}
```

> **Note:** Personal local settings merge with (not replace) the project settings.

---

## Skills: Slash Commands for Common Workflows

Skills are project-specific slash commands that encode knowledge Claude would otherwise have to figure out each time — like which feature flags to use, how to deploy to catbox, or what the test workflow looks like.

| Command | What it does |
|---------|-------------|
| `/build` | Build the project with various configurations (debug, release, feature flags) |
| `/test` | Run unit and integration tests |
| `/review` | Review recent changes for quality, correctness, and style |
| `/deploy` | Build, deploy to catbox (Optiplex 3040M), and restart the service |
| `/capture` | Capture a frame from the catbox camera and display it |
| `/tdd` | Test-driven development with red-green-refactor loop |
| `/wrapup` | End-of-session wrap-up — update beads issues, docs, and memory |

**Skills are opt-in.** You can always tell Claude what to do conversationally ("build in release mode with the web feature") and it will figure it out from the rules. Skills just make common workflows faster and more reliable.

**Skills and agents are separate concepts.** A skill is a slash command — you invoke it and Claude executes the workflow. An agent is a subprocess Claude spawns to do specialized work (coding, reviewing, testing). Skills don't require agents, and agents don't require skills, though some agents have access to certain skills internally.

**What each looks like in practice:**

A skill runs inline — you invoke it, Claude does the work directly:

```text
You:    /test
Claude: Running cargo fmt --check... ✓
        Running cargo clippy... ✓
        Running cargo test --lib... 77 passed, 0 failed
```

An agent spawns colored subprocesses that run in parallel with the main conversation:

```text
You:    Add a new detection model. Use agents.
Claude: [spawns coder]         → writes the code and tests
        [spawns code-reviewer] → reviews for bugs and style
        [spawns test-runner]   → builds, runs tests
```

---

## The Agent Pipeline: When and Why to Use It

The agent pipeline is Claude reviewing and testing Claude's own work before you ever see it.

### The Pipeline

```text
coder (green) ──→ code-reviewer (red) ──→ test-runner (yellow)
  writes code       reviews for bugs,       builds and runs
  and tests         style, conventions       the test suite
```

### Terminal Colors

When agents run, they appear color-coded in your terminal:

| Color | Agent | Role |
|-------|-------|------|
| Green | coder | Writes and refactors code |
| Red | code-reviewer | Reviews code, never modifies it |
| Yellow | test-runner | Builds and runs tests |

### When to Use the Pipeline

| Scenario | Recommendation |
|----------|---------------|
| One-line fix, typo | Skip the pipeline — just make the change |
| Small, well-defined change | "Use the coder." |
| New feature or significant refactor | Full pipeline: coder → code-reviewer → test-runner |
| Debugging a test failure | test-runner alone to reproduce, then coder to fix |
| Multi-file change across modules | Full pipeline with "Use agents." |

The pipeline is a preference for non-trivial work, not a gate on every interaction.

### How to Invoke Agents

Agents only run when you ask for them. To invoke the full pipeline, add **"Use agents."** to your request:

> "Add a new notification backend. Use agents."

Claude spawns the coder → code-reviewer → test-runner pipeline. Without that phrase, Claude does the work itself:

> "Add a new notification backend."

No agents — Claude writes the code directly in the main conversation.

You can also name specific agents if you only need part of the pipeline:

> "Run the test-runner."

For the full details on how subagents work — including background execution, persistent memory, hooks, and permission modes — see the [official subagents documentation](https://code.claude.com/docs/en/sub-agents).

---

## Plan Mode: Think Before You Build

Plan mode tells Claude to research and propose an approach before writing any code. This is useful when you're unsure how to approach a task, unfamiliar with the code being changed, or the change touches multiple files.

**How to activate:** Press `Shift+Tab` twice in the Claude Code input, type `/plan`, or just say "plan" in your prompt (e.g., "Plan how to add a new camera backend").

In plan mode, Claude explores the codebase, reads relevant files, and presents a written plan for your approval. You can annotate the plan, ask for changes, and iterate before any code is written. Once you approve, Claude exits plan mode and implements the plan.

**When to use it:**

| Scenario | Use plan mode? |
|----------|---------------|
| New feature spanning multiple modules | Yes — get alignment on approach before writing code |
| Unfamiliar area of the codebase | Yes — let Claude research first, then propose |
| Refactoring with multiple valid approaches | Yes — review trade-offs before committing to one |
| Bug fix where root cause is unclear | Yes — investigate before jumping to a fix |
| Simple change you could describe in one sentence | No — just do it directly |
| Renaming, formatting, typo fixes | No — plan mode adds overhead with no benefit |

**Iterating on plans:** You don't have to accept the first plan. Respond with feedback ("use a trait instead," "don't change the public API") and Claude will revise.

For more details, see the [official best practices](https://code.claude.com/docs/en/best-practices).

---

## Quick Reference

Slash commands are shortcuts, not requirements. You can always tell Claude what you want conversationally and it will figure out the right approach.

### Common Tasks

| I want to... | Do this |
|--------------|---------|
| Build the project | `/build` |
| Run tests | `/test` |
| Review my changes | `/review` |
| Deploy to catbox | `/deploy` |
| Capture a camera frame | `/capture` |
| Use the agent pipeline | Add "Use agents." to your request |
| Pre-approve a command | Add it to `.claude/settings.local.json` under `permissions.allow` |
| Add a project-wide rule | Create or edit a file in `.claude/rules/` and reference it from `CLAUDE.md` |
| Check open issues | `bd ready` or `bd list --status=open` |

### File Reference

| File | Who edits it | What it's for |
|------|-------------|---------------|
| `CLAUDE.md` | Team | Entry point — architecture, build commands, key details |
| `.claude/rules/*.md` | Team | Conventions Claude follows (Rust style, testing, git workflow) |
| `.claude/settings.json` | Team | Shared permissions |
| `.claude/settings.local.json` | You (gitignored) | Your personal permission overrides |
| `.claude/skills/*/` | Team | Slash-command definitions |
| `.claude/agents/*.md` | Team | Subagent role definitions |
| `config.example.toml` | Team | Reference configuration (copy to `config.toml` for local use) |
