"""Flask application factory.

This module provides the Flask application factory and
server startup utilities.
"""

import os
from typing import Optional, TYPE_CHECKING

from flask import Flask

from utils import Logger
from engine import MainEngine, init_engine_config

if TYPE_CHECKING:
    from .warehouse import WarehouseService


def create_app(config: Optional[dict] = None) -> Flask:
    """Create and configure Flask application.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    # Default configuration
    app.config.update(
        DEBUG=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True,
    )

    # Apply custom config
    if config:
        app.config.update(config)

    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp)

    # Add root health check
    @app.route("/")
    def root():
        return {"status": "ok", "service": "Task Management System"}

    @app.route("/health")
    def health():
        return {"status": "healthy"}

    # Initialize logging
    Logger.init()

    return app


def init_engine() -> MainEngine:
    """Initialize the main engine.

    Returns:
        Initialized MainEngine instance.
    """
    # Initialize engine config
    init_engine_config()

    # Get and initialize engine
    engine = MainEngine()
    engine.init()

    return engine


def init_warehouse(engine_port: int = 8555) -> Optional["WarehouseService"]:
    """Initialize the warehouse service.

    This function bridges the EngineConfig settings to WarehouseConfig
    and initializes the WarehouseService singleton.

    Args:
        engine_port: Port the API server is listening on.

    Returns:
        Initialized WarehouseService or None if disabled.
    """
    from engine import get_engine_config

    engine_config = get_engine_config()

    if not engine_config.warehouse_enabled:
        log = Logger.get_logging_method("WAREHOUSE")
        log("Warehouse integration disabled")
        return None

    from .warehouse import (
        WarehouseService,
        WarehouseConfig,
        MainEngineConnector,
        setup_warehouse_logger,
    )

    # Setup warehouse-specific logging
    setup_warehouse_logger(log_dir=engine_config.logs_dir)

    log = Logger.get_logging_method("WAREHOUSE")
    log("Initializing warehouse service...")

    # Create config from engine settings
    config = WarehouseConfig(
        enabled=engine_config.warehouse_enabled,
        sync_interval=engine_config.warehouse_sync_interval,
        engine_name=engine_config.service_name,
        engine_port=engine_port,
    )

    # Initialize service
    service = WarehouseService()
    service.init(config=config, connector=MainEngineConnector(config=config))

    return service


def run_server(
    host: str = "0.0.0.0",
    port: int = 8555,
    debug: bool = False,
    threaded: bool = True,
) -> None:
    """Run the Flask development server.

    Args:
        host: Host address to bind.
        port: Port number.
        debug: Enable debug mode.
        threaded: Enable threaded mode.
    """
    log = Logger.get_logging_method("API")
    log(f"Starting API server on {host}:{port}")

    # Initialize engine
    engine = init_engine()
    engine.start()

    # Initialize and start warehouse service
    warehouse = init_warehouse(engine_port=port)
    if warehouse:
        warehouse.start()

    # Create and run app
    app = create_app({"DEBUG": debug})

    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=threaded,
            use_reloader=False  # Disable reloader for multiprocessing
        )
    except KeyboardInterrupt:
        log("Shutting down...")
    finally:
        if warehouse:
            warehouse.stop()
        engine.stop()
        log("Server stopped")


def run_gunicorn(
    host: str = "0.0.0.0",
    port: int = 8555,
    workers: int = 1,
) -> None:
    """Run with Gunicorn WSGI server.

    Args:
        host: Host address to bind.
        port: Port number.
        workers: Number of worker processes.
    """
    try:
        import gunicorn.app.base

        class GunicornApp(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        # Initialize engine
        engine = init_engine()
        engine.start()

        # Initialize and start warehouse service
        warehouse = init_warehouse(engine_port=port)
        if warehouse:
            warehouse.start()

        options = {
            "bind": f"{host}:{port}",
            "workers": workers,
            "worker_class": "sync",
            "timeout": 120,
        }

        app = create_app()
        try:
            GunicornApp(app, options).run()
        finally:
            if warehouse:
                warehouse.stop()
            engine.stop()

    except ImportError:
        print("Gunicorn not installed. Install with: pip install gunicorn")
        print("Falling back to development server...")
        run_server(host, port)


# Application instance for WSGI servers
application = None
_warehouse_service: Optional["WarehouseService"] = None


def get_wsgi_app() -> Flask:
    """Get WSGI application instance.

    This is used by WSGI servers like Gunicorn.

    Returns:
        Flask application.
    """
    global application, _warehouse_service

    if application is None:
        # Initialize engine
        init_engine().start()

        # Initialize and start warehouse service
        _warehouse_service = init_warehouse()
        if _warehouse_service:
            _warehouse_service.start()

        application = create_app()

    return application
