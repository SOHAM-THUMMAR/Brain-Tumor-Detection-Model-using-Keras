import logging
from flask import Flask
from app.config import Config
from app.services.stats_service import ensure_directories_and_graphs
from app.services.model_service import load_model

logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    """
    Application Factory creating and configuring the Flask app instance.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize directories and sync evaluation graph images
    ensure_directories_and_graphs()

    # Pre-load singleton Keras model
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")

    # Register modular Blueprints
    from app.routes.main_routes import main_bp
    from app.routes.predict_routes import predict_bp
    from app.routes.stats_routes import stats_bp
    from app.routes.health_routes import health_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(health_bp)

    return app
