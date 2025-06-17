#!/usr/bin/env python3
"""Script to launch the AutoVisionAI Streamlit UI."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit UI."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Path to the Streamlit app
    app_path = project_root / "src" / "autovisionai" / "ui" / "app.py"

    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)

    # Launch Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
        "--browser.gatherUsageStats",
        "false",
    ]

    print("Starting AutoVisionAI UI...")
    print(f"Command: {' '.join(cmd)}")
    print("UI will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down UI server...")
    except subprocess.CalledProcessError as e:
        print(f"Error launching UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
