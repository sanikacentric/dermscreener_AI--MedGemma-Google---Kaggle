"""
DermScreen AI — Single-command launcher.

Usage:
    python run.py

What it does:
    1. Loads .env (DEMO_MODE, HF_TOKEN, etc.)
    2. Finds the first free port starting at 7861
    3. Kills any stale Python process holding that port (Windows-safe)
    4. Starts the Gradio app
    5. Opens http://localhost:<port> in your default browser
"""

import os
import sys
import socket
import subprocess
import webbrowser
import time
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── 1. Find a free port ───────────────────────────────────────────────────────

def find_free_port(start: int = 7861, end: int = 7880) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{end}.")


# ── 2. Kill any process holding a port (Windows + Unix) ───────────────────────

def free_port(port: int) -> None:
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                    pid = parts[4]
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True,
                                   capture_output=True)
                    log.info("Freed port %d (killed PID %s)", port, pid)
        else:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
            for pid in result.stdout.strip().split():
                subprocess.run(["kill", "-9", pid])
                log.info("Freed port %d (killed PID %s)", port, pid)
    except Exception as exc:
        log.warning("Could not free port %d: %s", port, exc)


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    preferred = int(os.getenv("PORT", "7861"))

    # Try to free the preferred port first
    free_port(preferred)
    time.sleep(1)

    port = find_free_port(start=preferred)
    log.info("Starting DermScreen AI on port %d", port)

    # Patch sys.path so `app` package is importable
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Lazy-import here so .env is already loaded
    import gradio as gr
    from app.main import build_app

    app = build_app()

    url = f"http://localhost:{port}"
    log.info("Opening browser at %s", url)

    # Open browser after a short delay so the server is up
    def _open():
        time.sleep(2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=_open, daemon=True).start()

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
        show_error=True,
    )


if __name__ == "__main__":
    main()
