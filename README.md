# 🤖 GenAI Personal Agent

A personal AI agent for your Mac — powered by **Claude Opus 4.6** with adaptive thinking, built on the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) and a [Gradio](https://gradio.app) web UI.

## What it can do

| Capability | How |
|---|---|
| Run shell commands & Python | `Bash` tool |
| Control macOS apps | `osascript` AppleScript via Bash |
| Read/write/edit files | `Read`, `Write`, `Edit` tools |
| Search your codebase | `Glob`, `Grep` tools |
| Search & read the web | `WebSearch`, `WebFetch` tools |
| Full browser automation | Playwright MCP (toggle in UI) |
| Data analysis | Runs pandas/matplotlib Python inline |
| Session memory | Resumes previous conversations |

## Quick start

### 1. Prerequisites

- Python 3.10+
- [Claude Code CLI](https://claude.ai/code) installed & authenticated (`claude --version`)
- Anthropic API key

### 2. Setup

```bash
cd ~/Desktop/GenAI
bash setup.sh
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

> If you're already authenticated via `claude login`, the Agent SDK may use that credential automatically — you may not need the `.env` file.

### 4. Launch

```bash
source .venv/bin/activate
python app.py
```

Opens automatically at **http://127.0.0.1:7860**

---

## Web Automation (Playwright)

Toggle "Web Automation" in the UI to let the agent control a browser.

**Requires Node.js:**
```bash
npm install -g @playwright/mcp
npx playwright install
```

---

## Example prompts

```
Analyze the CSV files in my Downloads and give me key statistics

Write a scikit-learn pipeline for classification, train it on iris data, and save it

Find all TODO comments in my Python projects

Open Safari and summarize the Claude.ai homepage

Create a matplotlib dashboard from my sales data

Check my Calendar and list this week's events

What are the top 10 largest files on my Mac?
```

---

## Architecture

```
app.py                    ← Gradio web UI + async streaming
  └── claude-agent-sdk    ← Wraps the Claude Code CLI subprocess
        └── Claude CLI    ← Calls Anthropic API with tools
              ├── Bash          (shell, Python, AppleScript)
              ├── Read/Write/Edit/Glob/Grep
              ├── WebSearch/WebFetch
              └── Playwright MCP (optional)
```

**Model:** `claude-opus-4-6` with `thinking: {type: "adaptive"}` — Claude decides when to think deeply.

**Session resumption:** The app maintains a `session_id` so follow-up messages have full context of what the agent already did.

---

## Customizing the agent

Edit the `SYSTEM_PROMPT` in [app.py](app.py) to give the agent domain-specific knowledge, extra instructions, or a different personality.

Add custom MCP servers in the `mcp_servers` dict inside the `respond()` function — e.g., a Postgres MCP for database access, or a Slack MCP for messaging.
