"""Serve the project page locally, without importing any ML packages."""
from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[1] / "site"
    if not (site / "index.html").is_file():
        parser.error("The site/index.html file is missing.")
    if not (site / "downloads" / "unisem-source.zip").is_file():
        subprocess.run([sys.executable, str(site.parent / "tools" / "package_release.py")], check=True)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"UniSEM project page: http://{args.host}:{args.port}", flush=True)
        print("Static research preview. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nUniSEM preview stopped.", flush=True)


if __name__ == "__main__":
    main()
