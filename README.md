# Local Ollama Assistant

A local-first Python assistant powered by Ollama and `qwen2.5-coder:7b`. It
includes:

- A terminal chat interface with streaming replies and local conversation history.
- An optional PySide6 desktop chat interface.
- A terminal coding agent that can inspect and edit approved workspace folders,
  search the web, read public GitHub information, and run a limited set of
  commands after approval.

## Requirements

- Python 3.9 or newer
- Ollama running locally
- The selected model installed in Ollama
- Textual for the terminal interface
- PySide6 for the optional desktop interface

Install the Python dependencies and download the default model:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
```

## Start the application

Run the launcher to choose the terminal or desktop interface:

```powershell
python main.py
```

Start an interface directly:

```powershell
python main.py --mode cli
python main.py --mode app
```

The terminal chat can also be launched with `python tui_app.py`, and the desktop
app with `python desktop_app.py`.

## Terminal chat

```powershell
python main.py --mode cli
```

The terminal interface starts the local Ollama server automatically if needed.
It opens a new, empty conversation on each launch; older conversations remain
available in the searchable history. Chat history and preferences are stored
locally in `%LOCALAPPDATA%\LocalOllamaAgent\chat_history.db`.

Use **Enter** to send a message, **Ctrl+N** to start a new conversation,
**Esc** to stop a reply, **Ctrl+Shift+C** to copy the latest answer, **Ctrl+O**
to open settings, and **Ctrl+Q** to quit. On exit, the app unloads the selected
model to release its VRAM.

Settings let you choose an installed Ollama model, select Turkish, English,
German, or French, manage folders available to the coding agent, personalize
the assistant, edit your profile, and clear saved conversations. The workspace
is always allowed; additional folder permissions apply only to the terminal
coding agent. The desktop chat has no file tools.

The prompt uses `$`. Enter a command by itself to:

- `--history` — search and select a saved conversation.
- `--settings` — open settings.
- `-clear` — start a new conversation without deleting history.
- `-undo` — preview and undo the most recent agent file change.
- `-deep` — enable model thinking for subsequent messages when supported.
- `-fast` — disable thinking.
- `--help` — show the command list.

Strings such as `-deep explain this` are treated as regular chat messages.

## Terminal coding agent

Start an interactive agent in the current folder:

```powershell
python local_ollama_agent.py --workspace .
```

Or give it a task:

```powershell
python local_ollama_agent.py --workspace . "Inspect this project and summarize its README"
```

The agent can access its selected workspace and additional folders granted in
the terminal chat settings. Paths are resolved and restricted to those allowed
roots. This application-level restriction does not grant or replace Windows
permissions and is not an operating-system sandbox.

For explicit file creation or editing requests, Ollama receives native file
tools. If the model does not call a tool, the agent retries and reports that
the request was not completed rather than claiming a file was saved. Before a
file is written or changed, the agent displays a diff and requires approval.
The latest agent edit can be undone with `-undo` during the current session.

Terminal commands are not an unrestricted shell. `run_shell` allows only
`python -m unittest discover` and read-only Git commands (`status`, `diff`,
`log`, and `branch --show-current`); other commands and shell operators are
rejected. Test code still runs with the current Windows account's permissions,
so this allowlist is not an operating-system sandbox.

Git write tools work only when the workspace itself is the repository root.
File edits and commits are blocked on `main`/`master` until a separate feature
branch is created. Branch creation requires a clean working tree. Commits stage
only explicitly named files and require approval.

## Desktop chat

```powershell
python main.py --mode app
```

The desktop app uses PySide6, starts the local Ollama server if needed, and
checks that the configured model is installed. Each launch opens a fresh
conversation while preserving earlier chats in the searchable sidebar.
Conversations and preferences remain on this computer. Use **Enter** to send
messages.

The sidebar displays CPU and system memory usage. NVIDIA GPU utilization and
VRAM are shown when `nvidia-smi` is available; Ollama model VRAM is read from
its local process API. The desktop interface is for conversation; use the
terminal coding agent for file, web, GitHub, and terminal tools.

## Web and GitHub

Web search and public GitHub repository and issue lookups use HTTPS and do not
require a GitHub login. Publishing the project requires a GitHub token in the
`GITHUB_TOKEN` environment variable. Use a token with only the permissions
needed to write to the target repository. Never paste the token into a chat,
source file, or commit.

On Windows, set a token for the current PowerShell session without putting it
in shell history:

```powershell
$secureToken = Read-Host "GitHub token" -AsSecureString
$env:GITHUB_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
```

The token is read from the environment and is not printed by the application.
The `github_publish_project` tool uploads this project's fixed file manifest
only after explicit approval. Existing repositories must have the requested
visibility.

GitHub agent commits include a Copilot `Co-authored-by` trailer. Reading files
that may contain secrets, such as `.env`, token or credential files, and
private-key files, requires explicit approval.

## Configuration

Command-line options:

```powershell
python local_ollama_agent.py --model qwen2.5-coder:7b --ollama-url http://localhost:11434 --max-turns 20 --workspace .
```

The `OLLAMA_MODEL`, `OLLAMA_BASE`, and `GITHUB_TOKEN` environment variables can
be used where applicable.

## Tests

Run the test suite from the project directory:

```powershell
python -m unittest discover -s tests -v
```

