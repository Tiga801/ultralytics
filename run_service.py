#!/usr/bin/env python3
"""Main service entry point.

This script provides the entry point for running the task management
service. It initializes the engine and starts the Flask API server.

Usage:
    # Run with default settings
    python run_service.py

    # Run with custom host and port
    python run_service.py --host 0.0.0.0 --port 8666

    # Run in debug mode
    python run_service.py --debug

    # Run with Gunicorn (production)
    python run_service.py --production
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Task Management Service"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8666,
        help="Port number (default: 8666)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run with Gunicorn for production"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Gunicorn workers (default: 1)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize logging
    Logger.init()
    log = Logger.get_logging_method("MAIN")

    log("=" * 60)
    log("Task Management Service")
    log("=" * 60)

    # Import solutions to register task types
    log("Loading solution modules...")
    try:
        import solutions  # useful, register task type
        from task import TaskRegistry
        registered = TaskRegistry.get_supported_types()
        log(f"Registered task types: {registered}")
    except ImportError as e:
        log(f"Warning: Could not load solutions: {e}")

    # Start the service
    if args.production:
        log(f"Starting production server on {args.host}:{args.port}")
        log(f"Workers: {args.workers}")
        from api import run_gunicorn
        run_gunicorn(
            host=args.host,
            port=args.port,
            workers=args.workers
        )
    else:
        log(f"Starting development server on {args.host}:{args.port}")
        log(f"Debug mode: {args.debug}")
        from api import run_server
        run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
