"""
GenAI Personal Agent
====================
Dual-provider agentic chatbot:
  - Claude (Anthropic) — full Agent SDK with tools, Bash, web, Playwright
  - Ollama (local)     — streaming chat, 100% offline, no API costs

Usage:
    python app.py  →  http://127.0.0.1:7860
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

DEFAULT_HOME = str(Path.home())

# ── Prompts ────────────────────────────────────────────────────────────────────

CLAUDE_SYSTEM_PROMPT = """You are a powerful AI assistant running directly on this Mac.
You have broad tool access and should take initiative to fully complete tasks.

## Tools at your disposal

**Bash** — run any shell command, Python script, or AppleScript:
- Execute Python: `python3 -c "..."` or write and run a script
- macOS app control via `osascript`:
  - Open an app:     `open -a "Safari"`
  - Spotify:         `osascript -e 'tell application "Spotify" to play'`
  - iMessage:        `osascript -e 'tell app "Messages" to send "hi" to buddy "+15550100"'`
  - Screenshot:      `screencapture -x ~/Desktop/shot.png`
  - Clipboard:       `osascript -e 'get the clipboard'`
- System info: `top -l 1 | head -20`, `df -h`, `vm_stat`

**Read / Write / Edit / Glob / Grep** — manage and search files

**WebSearch / WebFetch** — search and read the web

**Playwright MCP** (when enabled) — full browser automation

## Data analysis
Prefer running Python inline for data tasks:
```bash
python3 << 'EOF'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
print(df.describe())
plt.savefig("out.png")
EOF
```

## Guidelines
- Use absolute paths · show outputs · ask before destructive actions
"""

OLLAMA_SYSTEM_PROMPT = """You are a helpful, concise AI assistant.
You are running fully locally — no internet access, no external tools.
Answer clearly and directly. Use markdown and code blocks when helpful."""

# ── Quick prompts ──────────────────────────────────────────────────────────────

QUICK_PROMPTS = [
    ("📊 Analyze data",       "Analyze any CSV or data files in my Downloads folder and give me key insights"),
    ("🖥️ System status",     "Show me my Mac's CPU, memory, disk usage, and top processes"),
    ("🐍 Write data script",  "Write a Python script using pandas and matplotlib: load a CSV, clean it, make 3 visualizations"),
    ("🌐 Browse & summarize", "Search for the latest AI and data science news, summarize the top 5 stories"),
    ("📅 Calendar check",     "Use AppleScript to list my upcoming Calendar events"),
    ("🔍 Find large files",   "Find the 20 largest files on my Mac (skip system dirs) and show sizes"),
    ("📱 Draft iMessage",     "Help me compose and optionally send an iMessage"),
    ("🤖 ML pipeline",        "Create a scikit-learn pipeline: load data, preprocess, train, evaluate, save"),
    ("🗂️ Desktop organizer", "List my Desktop contents and suggest an organization strategy"),
    ("🔧 Fix Python env",     "Check my Python environment and fix any common issues"),
]

# ── Ollama helpers ─────────────────────────────────────────────────────────────


def list_ollama_models() -> list[str]:
    """Return locally available Ollama models, or a fallback list."""
    try:
        import ollama  # type: ignore
        models = ollama.list()
        names = [m["name"] for m in models.get("models", [])]
        return names if names else ["llama3.2", "mistral", "qwen2.5"]
    except Exception:
        return ["llama3.2", "mistral", "qwen2.5", "codellama", "phi3"]


def history_to_ollama(history: list, system: str) -> list[dict]:
    """Convert Gradio history messages to Ollama message format."""
    messages = [{"role": "system", "content": system}]
    for msg in history:
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


# ── App builder ────────────────────────────────────────────────────────────────


def build_app():
    with gr.Blocks(
        title="GenAI Personal Agent",
    ) as app:

        # ── State ──
        session_state = gr.State(None)   # Claude session ID for resumption

        # ── Header ──
        gr.Markdown(
            """
            # 🤖 GenAI Personal Agent
            *Claude Opus 4.6 (agentic + tools) · Ollama (local, offline)*
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
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask me anything…",
                        show_label=False,
                        lines=1,
                        scale=6,
                        interactive=True,
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

                    provider = gr.Radio(
                        label="Provider",
                        choices=["🤖 Claude (Agentic)", "🦙 Ollama (Local)"],
                        value="🤖 Claude (Agentic)",
                    )

                    # ── Claude settings ──
                    with gr.Group() as claude_settings:
                        claude_model = gr.Dropdown(
                            label="Claude Model",
                            choices=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
                            value="claude-opus-4-6",
                        )
                        permission_mode = gr.Dropdown(
                            label="Permission Mode",
                            choices=["acceptEdits", "default", "bypassPermissions"],
                            value="acceptEdits",
                            info="acceptEdits = auto-approve file writes",
                        )
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

                    # ── Ollama settings ──
                    with gr.Group(visible=False) as ollama_settings:
                        ollama_models = list_ollama_models()
                        ollama_model = gr.Dropdown(
                            label="Ollama Model",
                            choices=ollama_models,
                            value=ollama_models[0] if ollama_models else "llama3.2",
                            allow_custom_value=True,
                            info="Run `ollama pull <model>` to download",
                        )
                        ollama_refresh = gr.Button("🔄 Refresh model list", size="sm")

                with gr.Accordion("💡 Quick Prompts", open=True):
                    for label, prompt_text in QUICK_PROMPTS:
                        gr.Button(label, size="sm", elem_classes="quick-btn").click(
                            fn=lambda p=prompt_text: p,
                            outputs=user_input,
                        )

        # ── Provider switcher ──────────────────────────────────────────────────

        def switch_provider(prov):
            is_ollama = "Ollama" in prov
            return gr.update(visible=not is_ollama), gr.update(visible=is_ollama)

        provider.change(
            fn=switch_provider,
            inputs=provider,
            outputs=[claude_settings, ollama_settings],
        )

        def refresh_models():
            models = list_ollama_models()
            return gr.update(choices=models, value=models[0] if models else "llama3.2")

        ollama_refresh.click(fn=refresh_models, outputs=ollama_model)

        # ── Claude agent ───────────────────────────────────────────────────────

        async def respond_claude(message, history, session_id, use_playwright, cwd, model, perm_mode):
            try:
                from claude_agent_sdk import (  # type: ignore
                    query, ClaudeAgentOptions, CLINotFoundError, CLIConnectionError,
                )
            except ImportError:
                history[-1][1] = "❌ `claude-agent-sdk` not installed.\n```bash\npip install claude-agent-sdk\n```"
                yield history, session_id, "Error"
                return

            tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
            mcp_servers = {}
            if use_playwright:
                mcp_servers["playwright"] = {"command": "npx", "args": ["@playwright/mcp@latest"]}

            opts = dict(
                model=model,
                cwd=cwd or DEFAULT_HOME,
                allowed_tools=tools,
                permission_mode=perm_mode,
                system_prompt=CLAUDE_SYSTEM_PROMPT,
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
                    subtype = getattr(msg, "subtype", None)
                    content = getattr(msg, "content", None)

                    # Result message — has subtype='success' and a result field
                    if subtype == "success":
                        new_session_id = getattr(msg, "session_id", new_session_id)
                        result_text = getattr(msg, "result", result_text)
                        history[-1]["content"] = result_text
                        yield history, new_session_id, "✅ Done"

                    # Assistant message — content list contains TextBlock objects
                    elif isinstance(content, list) and content and hasattr(content[0], "text"):
                        parts = [b.text for b in content if hasattr(b, "text")]
                        streamed = "".join(parts)
                        if streamed:
                            result_text = streamed
                            history[-1]["content"] = result_text
                            yield history, new_session_id, "💬 Responding…"

                    # Tool result — content list contains ToolResultBlock objects
                    elif isinstance(content, list) and content and hasattr(content[0], "tool_use_id"):
                        yield history, new_session_id, "🔧 Running tool…"

                if history[-1]["content"] == "":
                    history[-1]["content"] = "✅ Task complete."
                    yield history, new_session_id, "✅ Done"

            except CLINotFoundError:
                history[-1]["content"] = "❌ Claude Code CLI not found. Make sure `claude` is on your PATH."
                yield history, new_session_id, "Error"
            except CLIConnectionError as e:
                history[-1]["content"] = f"❌ Connection error: {e}"
                yield history, new_session_id, "Error"
            except Exception as e:
                history[-1]["content"] = f"❌ {type(e).__name__}: {e}"
                yield history, new_session_id, "Error"

        # ── Ollama chat ────────────────────────────────────────────────────────

        async def respond_ollama(message, history, model_name):
            try:
                from ollama import AsyncClient  # type: ignore
            except ImportError:
                history[-1]["content"] = "❌ `ollama` not installed.\n```bash\npip install ollama\n```"
                yield history, None, "Error"
                return

            messages = history_to_ollama(history[:-2], OLLAMA_SYSTEM_PROMPT)
            messages.append({"role": "user", "content": message})

            response_text = ""
            try:
                client = AsyncClient()
                async for chunk in await client.chat(
                    model=model_name, messages=messages, stream=True
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    response_text += delta
                    history[-1]["content"] = response_text
                    yield history, None, "💬 Streaming…"

                yield history, None, "✅ Done"

            except Exception as e:
                err = str(e)
                if "connection" in err.lower() or "refused" in err.lower():
                    history[-1]["content"] = (
                        "❌ Cannot connect to Ollama.\n\n"
                        "Make sure Ollama is running:\n```bash\nollama serve\n```"
                    )
                else:
                    history[-1]["content"] = f"❌ {type(e).__name__}: {e}"
                yield history, None, "Error"

        # ── Router ─────────────────────────────────────────────────────────────

        async def respond(
            message, history, session_id,
            prov, use_playwright, cwd, c_model, perm_mode, o_model,
        ):
            if not message or not message.strip():
                yield history, session_id, "Ready"
                return

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, session_id, "🤔 Thinking…"

            if "Ollama" in prov:
                async for state in respond_ollama(message, history, o_model):
                    yield state
            else:
                async for state in respond_claude(
                    message, history, session_id, use_playwright, cwd, c_model, perm_mode
                ):
                    yield state

        def new_session():
            return [], None, "🆕 New session started"

        # ── Event wiring ──────────────────────────────────────────────────────

        agent_inputs = [
            user_input, chatbot, session_state,
            provider, playwright_toggle, cwd_input,
            claude_model, permission_mode, ollama_model,
        ]
        agent_outputs = [chatbot, session_state, status_box]

        send_btn.click(fn=respond, inputs=agent_inputs, outputs=agent_outputs).then(
            fn=lambda: "", outputs=user_input
        )
        user_input.submit(fn=respond, inputs=agent_inputs, outputs=agent_outputs).then(
            fn=lambda: "", outputs=user_input
        )
        clear_btn.click(fn=new_session, outputs=agent_outputs)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
        footer { display: none !important; }
        #chatbot .message { font-size: 0.95rem; }
        .quick-btn { text-align: left !important; }
        """,
        show_error=True,
        inbrowser=True,
    )
