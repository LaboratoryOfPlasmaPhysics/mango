import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="mango", description="MANGO dataset service")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the MANGO API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve_parser.add_argument("--data-dir", default=None, help="Path to Hive-partitioned Parquet directory (overrides MANGO_DATA_DIR)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _serve(args)


def _serve(args):
    import os

    import uvicorn

    if args.data_dir:
        os.environ["MANGO_DATA_DIR"] = args.data_dir

    uvicorn.run("mango.main:app", host=args.host, port=args.port, reload=args.reload)
