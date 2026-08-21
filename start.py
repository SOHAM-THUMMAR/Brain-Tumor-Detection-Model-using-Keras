import os
import sys
import time
import threading
import webbrowser
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("start")


def open_browser(url, delay=1.5):
    """Waits for server initialization then opens the web browser."""
    time.sleep(delay)
    logger.info(f"Opening web app in default browser: {url}")
    webbrowser.open(url)


def main():
    logger.info("Starting NeuroScan AI Brain Tumor Detection Web Application...")

    # Add workspace directory to Python path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        from app import create_app
        app = create_app()
    except Exception as e:
        logger.error(f"Failed to initialize Flask application: {e}", exc_info=True)
        sys.exit(1)

    url = "http://127.0.0.1:5000"

    # Launch browser thread
    threading.Thread(target=open_browser, args=(url, 1.5), daemon=True).start()

    print("\n" + "=" * 60)
    print(" 🧠 NEUROSCAN AI - BRAIN TUMOR DETECTION WEB SERVER")
    print(f" Server active at: {url}")
    print(" Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
