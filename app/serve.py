"""Serve the static interface locally; predictions use the saved-model export."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parent))
    print(f"PhaseGuard: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
