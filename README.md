# CoderAI

A real coding AI agent for your terminal — uses the OpenAI API (or any OpenAI-compatible endpoint) with tool calling, like a mini OpenCode/ChatGPT-for-code.

## Features
- Agentic loop: the model decides when to call tools and iterates until done
- Tools: `run_command`, `read_file`, `write_file`, `edit_file`, `list_dir`, `search`
- Works with OpenAI, or any compatible endpoint (set `OPENAI_BASE_URL`)

## Setup
```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
# optional:
$env:CODERAI_MODEL = "gpt-4o-mini"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

## Run
```powershell
python coderai.py
```

Commands: `/clear` reset conversation, `/exit` quit.

## How it works
1. You type a task.
2. The model reasons and calls tools (read/write/edit files, run shell).
3. Tool results are fed back to the model.
4. Loop until the model answers without more tool calls.
