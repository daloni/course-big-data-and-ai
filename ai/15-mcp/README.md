# MCP with uv

```sh
pip install uv
uv init
uv add mcp
uv add mcp[cli]
```

## Debug tool
```sh
pip install mcp[cli]
mcp dev main.py
```

## Install 5ire app

URL: https://5ire.app/

Run app:

```sh
chmod +x 5ire-0.9.9-x86_64.AppImage
./5ire-0.9.9-x86_64.AppImage
```

## Config mcp

Add new tool with the command:

```sh
~/.venv/bin/uv run --with mcp[cli] mcp run ~/main.py
```
