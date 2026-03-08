#!/usr/bin/env bash
# GenAI Agent — one-shot setup
set -e

echo "🤖 Setting up GenAI Agent..."

# 1. Python virtual environment
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Python dependencies
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Python dependencies installed."

# 3. Node.js check (needed only for Playwright web automation)
if command -v node &>/dev/null; then
    echo "✅ Node.js found: $(node --version)"
    echo "   To enable web automation: npm install -g @playwright/mcp"
else
    echo "⚠️  Node.js not found — web automation (Playwright) won't work."
    echo "   Install Node.js from https://nodejs.org or via Homebrew: brew install node"
fi

# 4. ANTHROPIC_API_KEY check
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ -f .env ]; then
        echo "✅ .env file found — API key will be loaded from it."
    else
        echo ""
        echo "⚠️  ANTHROPIC_API_KEY not set."
        echo "   Copy .env.example → .env and add your key:"
        echo "   cp .env.example .env && open .env"
    fi
else
    echo "✅ ANTHROPIC_API_KEY is set."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  To launch the agent:"
echo "    source .venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Then open: http://127.0.0.1:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
