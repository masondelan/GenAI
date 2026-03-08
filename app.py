"""
GenAI Personal Agent
====================
A Gradio-powered agentic chatbot using Claude Opus 4.6 + Agent SDK.

Capabilities:
  - Bash / shell execution (runs Python, scripts, AppleScript for macOS control)
  - File management (read, write, edit)
  - Web search & fetch
  - Playwright browser automation (optional, requires Node.js)
  - Session memory (resumes previous conversations)

Usage:
    python app.py
    Then open: http://127.0.0.1:7860
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

DEFAULT_HOME = str(Path.home())

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a powerful AI assistant running directly on this Mac.
You have broad tool access and should take initiative to fully complete tasks.

## Tools at your disposal

**Bash** — run any shell command, Python script, or AppleScript:
- Execute Python: `python3 script.py` or inline: `python3 -c "..."`
- AppleScript for native macOS app control via `osascript`:
  - Open an app:       `open -a "Safari"`
  - Control Spotify:   `osascript -e 'tell application "Spotify" to play'`
  - iMessage:          `osascript -e 'tell app "Messages" to send "hi" to buddy "+15555550100"'`
  - Create calendar event via AppleScript
  - Keystroke/type:    `osascript -e 'tell app "System Events" to keystroke "hello"'`
  - Screenshot:        `screencapture -x ~/Desktop/shot.png`
  - Clipboard:         `osascript -e 'get the clipboard'`
- System info: `top -l 1 | head -20`, `df -h`, `vm_stat`, `system_profiler SPHardwareDataType`
- Network: `curl`, `wget`, `ping`, `netstat`

**Read / Write / Edit / Glob / Grep** — manage files and search content

**WebSearch / WebFetch** — search the web and read pages

**Playwright MCP** (when enabled) — full browser automation:
navigate, click, fill forms, screenshot, scrape JS-rendered pages

## Data analysis best practice
Prefer running Python directly for data tasks:
```bash
python3 << 'EOF'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
print(df.describe())
plt.figure(); df["col"].hist(); plt.savefig("out.png")
EOF
```

## Guidelines
- Always use absolute paths
- Show command output when informative
- Ask before irreversible/destructive actions
- Prefer concise, direct responses with code blocks for commands
- Be proactive: if you need to install a package, do it
"""

# ── Quick prompts ──────────────────────────────────────────────────────────────

QUICK_PROMPTS = [
    ("📊 Analyze data", "Analyze any CSV or data files in my Downloads folder and give me key insights with statistics"),
    ("🖥️ System status", "Show me a full summary of my Mac: CPU, memory, disk usage, top processes, and battery"),
    ("🐍 Write data script", "Write a Python script using pandas and matplotlib that loads a CSV, cleans it, and creates 3 insightful visualizations"),
    ("🌐 Browse & summarize", "Search for the latest AI and data science news and summarize the top 5 stories with links"),
    ("📅 Calendar check", "Use AppleScript to check my upcoming Calendar events and list them"),
    ("🔍 Find large files", "Find the 20 largest files on my Mac (excluding system dirs) and show sizes"),
    ("📱 Draft iMessage", "Help me compose and optionally send an iMessage to a contact"),
    ("🗂️ Desktop organizer", "List everything on my Desktop and suggest (or implement) an organization strategy"),
    ("🔧 Fix Python env", "Check my Python environment, list installed packages, and fix any common issues"),
    ("🤖 ML pipeline", "Create a complete scikit-learn pipeline: load data, preprocess, train a model, evaluate, and save it"),
]

# ── App builder ────────────────────────────────────────────────────────────────


def build_app():
    with gr.Blocks(
        title="GenAI Personal Agent",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
        footer { display: none !important; }
        #chatbot .message { font-size: 0.95rem; }
        .quick-btn { text-align: left !important; }
        """,
    ) as app:

        # ── Persistent state ──
        session_state = gr.State(None)   # Stores session_id for resumption

        # ── Header ──
        gr.Markdown(
            """
            # 🤖 GenAI Personal Agent
            *Claude Opus 4.6 · Adaptive Thinking · macOS Native · Web Automation*
            """
        )

        # ── Layout ──
        with gr.Row(equal_height=False):

            # ── Left: Chat ──
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=540,
                    show_label=False,
                    render_markdown=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask me to do anything on your Mac…",
                        show_label=False,
                        lines=1,
                        scale=6,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)

                with gr.Row():
                    clear_btn = gr.Button("🔄 New Session", size="sm", scale=1)
                    status_box = gr.Textbox(
                        value="Ready",
                        show_label=False,
                        interactive=False,
                        scale=5,
                        max_lines=1,
                    )

            # ── Right: Controls ──
            with gr.Column(scale=2):

                with gr.Accordion("⚙️ Settings", open=True):
                    playwright_toggle = gr.Checkbox(
                        label="🌐 Web Automation (Playwright MCP)",
                        value=False,
                        info="Requires Node.js. Install: npm i -g @playwright/mcp",
                    )
                    cwd_input = gr.Textbox(
                        label="📁 Working Directory",
                        value=DEFAULT_HOME,
                        lines=1,
                    )
                    model_choice = gr.Dropdown(
                        label="Model",
                        choices=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
                        value="claude-opus-4-6",
                    )
                    permission_mode = gr.Dropdown(
                        label="Permission Mode",
                        choices=["acceptEdits", "default", "bypassPermissions"],
                        value="acceptEdits",
                        info="acceptEdits = auto-approve file writes",
                    )

                with gr.Accordion("💡 Quick Prompts", open=True):
                    for label, prompt_text in QUICK_PROMPTS:
                        gr.Button(label, size="sm", elem_classes="quick-btn").click(
                            fn=lambda p=prompt_text: p,
                            outputs=user_input,
                        )

        # ── Agent logic ───────────────────────────────────────────────────────

        async def respond(message, history, session_id, use_playwright, cwd, model, perm_mode):
            if not message or not message.strip():
                yield history, session_id, "Ready"
                return

            # Import here so the app still launches even if SDK isn't installed yet
            try:
                from claude_agent_sdk import (  # type: ignore
                    query,
                    ClaudeAgentOptions,
                    CLINotFoundError,
                    CLIConnectionError,
                )
            except ImportError:
                history = history + [
                    [
                        message,
                        "❌ `claude-agent-sdk` not installed.\n\nRun:\n```bash\npip install claude-agent-sdk\n```",
                    ]
                ]
                yield history, session_id, "Error — missing dependency"
                return

            history = history + [[message, ""]]
            yield history, session_id, "🤔 Thinking…"

            tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]

            mcp_servers = {}
            if use_playwright:
                mcp_servers["playwright"] = {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                }

            opts = dict(
                model=model,
                cwd=cwd or DEFAULT_HOME,
                allowed_tools=tools,
                permission_mode=perm_mode,
                system_prompt=SYSTEM_PROMPT,
                max_turns=50,
                mcp_servers=mcp_servers,
            )
            if session_id:
                opts["resume"] = session_id

            options = ClaudeAgentOptions(**opts)

            new_session_id = session_id
            result_text = ""

            try:
                async for msg in query(prompt=message, options=options):
                    mtype = getattr(msg, "type", None)

                    if mtype == "system":
                        if getattr(msg, "subtype", "") == "init":
                            new_session_id = getattr(msg, "session_id", session_id)

                    elif mtype == "result":
                        result_text = getattr(msg, "result", "")
                        history[-1][1] = result_text
                        yield history, new_session_id, "✅ Done"

                    elif mtype == "assistant":
                        # Stream partial text as it arrives
                        content = getattr(msg, "content", [])
                        if isinstance(content, list):
                            parts = [getattr(b, "text", "") for b in content if hasattr(b, "text")]
                            streamed = "".join(parts)
                            if streamed:
                                result_text = streamed
                                history[-1][1] = result_text
                                yield history, new_session_id, "💬 Responding…"

                    elif mtype in ("tool_use", "tool_call", "tool"):
                        tool_name = getattr(msg, "name", "tool")
                        yield history, new_session_id, f"🔧 Using: {tool_name}…"

                # Fallback — ensure the chat bubble is never left empty
                if history[-1][1] == "":
                    history[-1][1] = "✅ Task complete."
                    yield history, new_session_id, "✅ Done"

            except CLINotFoundError:
                history[-1][1] = (
                    "❌ **Claude Code CLI not found.**\n\n"
                    "Make sure `claude` is on your PATH and you're authenticated."
                )
                yield history, new_session_id, "Error"

            except CLIConnectionError as e:
                history[-1][1] = f"❌ Connection error: {e}"
                yield history, new_session_id, "Error"

            except Exception as e:
                history[-1][1] = f"❌ {type(e).__name__}: {e}"
                yield history, new_session_id, "Error"

        def new_session():
            return [], None, "🆕 New session started"

        # ── Event wiring ──────────────────────────────────────────────────────

        agent_inputs = [user_input, chatbot, session_state, playwright_toggle, cwd_input, model_choice, permission_mode]
        agent_outputs = [chatbot, session_state, status_box]

        send_btn.click(fn=respond, inputs=agent_inputs, outputs=agent_outputs).then(
            fn=lambda: "", outputs=user_input
        )
        user_input.submit(fn=respond, inputs=agent_inputs, outputs=agent_outputs).then(
            fn=lambda: "", outputs=user_input
        )
        clear_btn.click(fn=new_session, outputs=agent_outputs[:3])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,   # opens browser tab automatically
    )
