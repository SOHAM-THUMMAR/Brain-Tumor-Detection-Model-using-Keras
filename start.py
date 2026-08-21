import os
import sys
import time
import subprocess
import threading
import webbrowser
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("start")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")


def get_venv_python():
    """Returns the path to the Python executable inside the venv folder."""
    if sys.platform == "win32":
        scripts_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        bin_python = os.path.join(VENV_DIR, "bin", "python.exe")
        if os.path.exists(scripts_python):
            return scripts_python
        if os.path.exists(bin_python):
            return bin_python
        return scripts_python
    else:
        return os.path.join(VENV_DIR, "bin", "python")


def get_venv_pip():
    """Returns the path to the Pip executable inside the venv folder."""
    if sys.platform == "win32":
        scripts_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        bin_pip = os.path.join(VENV_DIR, "bin", "pip.exe")
        if os.path.exists(scripts_pip):
            return scripts_pip
        if os.path.exists(bin_pip):
            return bin_pip
        return scripts_pip
    else:
        return os.path.join(VENV_DIR, "bin", "pip")


def ensure_venv():
    """Creates virtualenv if missing, and re-executes script using venv python."""
    venv_python = get_venv_python()

    # Step 1: Create venv if missing
    if not os.path.exists(VENV_DIR) or not os.path.exists(venv_python):
        logger.info("Virtual environment 'venv' not found. Creating virtual environment...")
        try:
            import venv
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(VENV_DIR)
        except Exception as e:
            logger.warning(f"Built-in venv builder failed ({e}). Falling back to subprocess...")
            subprocess.check_call([sys.executable, "-m", "venv", "venv"], cwd=BASE_DIR)
        logger.info("Virtual environment created successfully.")

    # Step 2: Check if running inside the venv
    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(venv_python)

    # Compare paths (case-insensitive for Windows)
    if os.path.normcase(current_python) != os.path.normcase(target_python):
        logger.info(f"Re-executing script inside virtual environment ({target_python})...")
        cmd = [target_python, __file__] + sys.argv[1:]
        result = subprocess.run(cmd, cwd=BASE_DIR)
        sys.exit(result.returncode)


def ensure_dependencies():
    """Checks required packages and automatically installs requirements.txt if missing."""
    required_modules = ["flask", "tensorflow", "numpy", "PIL", "cv2", "matplotlib", "docx"]
    missing = False

    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing = True
            logger.info(f"Missing dependency: {mod}")
            break

    if missing:
        req_file = os.path.join(BASE_DIR, "requirements.txt")
        if os.path.exists(req_file):
            logger.info("Installing dependencies from requirements.txt...")
            venv_python = get_venv_python()
            subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_file], cwd=BASE_DIR)
            logger.info("All dependencies installed successfully.")
        else:
            logger.error("requirements.txt not found. Cannot auto-install dependencies.")


def open_browser(url, delay=1.5):
    """Waits for server initialization then opens default browser."""
    time.sleep(delay)
    logger.info(f"Opening web app in default browser: {url}")
    webbrowser.open(url)


def main():
    # 1. Ensure virtual environment exists and is active
    ensure_venv()

    # 2. Ensure all requirements are installed
    ensure_dependencies()

    logger.info("Starting NeuroScan AI Brain Tumor Detection Web Application...")

    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    try:
        from app.__init__ import create_app
        app = create_app()
    except Exception as e:
        logger.error(f"Failed to initialize Flask application: {e}", exc_info=True)
        sys.exit(1)

    url = "http://127.0.0.1:5000"

    # Launch browser thread
    threading.Thread(target=open_browser, args=(url, 1.5), daemon=True).start()

    print("\n" + "=" * 60)
    print(" NEUROSCAN AI - BRAIN TUMOR DETECTION WEB SERVER")
    print(f" Server active at: {url}")
    print(" Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")


    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
