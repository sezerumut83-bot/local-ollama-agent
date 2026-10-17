# Local Ollama Coding Agent

A standalone project directory for the local agent.

A small, local-first coding agent that uses Ollama's chat API. The default model
is `qwen2.5-coder:7b`. It can inspect and edit files inside a workspace, search
the web, read public GitHub repository information, and run terminal commands
only after you explicitly approve each command.

## Requirements

- Python 3.9 or newer
- Ollama running locally
- The model installed in Ollama

```powershell
ollama pull qwen2.5-coder:7b
python -m pip install -r requirements.txt
```

## Run

From the project folder:

```powershell
python local_ollama_agent.py --workspace . "Bu klasörü incele ve bir README özeti hazırla"
```

Or start interactively:

```powershell
python local_ollama_agent.py --workspace .
```

The agent is limited to the selected workspace for file operations. Every
terminal command is shown with its working directory and runs only when you
type `evet`; pressing Enter or entering anything else cancels it. GitHub
repository creation also asks for separate confirmation.

## Web and GitHub

Web search and public GitHub repository/issue lookups use HTTPS and do not need
a GitHub login. To let the agent create a repository in your account or publish
this standalone project, set a
`GITHUB_TOKEN` environment variable locally. Use a token with only the
permissions required to create a repository in your account and write its
contents. Never paste the token into a chat, task, source file, or commit.

On Windows, you can set the token for the current PowerShell window with:

```powershell
$secureToken = Read-Host "GitHub token" -AsSecureString
$env:GITHUB_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
```

The token is read by the process from the environment and is not printed by the
agent. Use the `github_publish_project` tool to create the repository and upload
only this project's `.gitignore`, README, requirements, source, and tests. The
agent displays the target and file list and proceeds only after you type `evet`.

## Configuration

```powershell
python local_ollama_agent.py --model qwen2.5-coder:7b --ollama-url http://localhost:11434 --max-turns 20 --workspace .
```

`OLLAMA_MODEL`, `OLLAMA_BASE`, and `GITHUB_TOKEN` environment variables can be
used instead of command-line options where applicable.
