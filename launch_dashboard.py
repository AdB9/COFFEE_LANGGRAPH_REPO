#!/usr/bin/env python3
"""
Alpha Seeker Dashboard Launcher
Provides multiple dashboard options based on complexity needs
"""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch Alpha Seeker Dashboard")
    parser.add_argument(
        "--version",
        choices=["simple", "full"],
        default="simple",
        help="Dashboard version to launch (default: simple)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected.")
        print("Please activate your virtual environment first:")
        print("source .venv/bin/activate")
        print()
    
    # Choose dashboard file
    if args.version == "simple":
        dashboard_file = "simple_dashboard.py"
        print("🚀 Launching Simple Alpha Seeker Dashboard...")
    else:
        dashboard_file = "alpha_dashboard.py"
        print("🚀 Launching Alpha Seeker Dashboard (Single Page)...")
    
    print(f"📍 Dashboard will open at http://localhost:{args.port}")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_file,
            "--server.port", str(args.port),
            "--server.address", "localhost",
            "--theme.base", "light",
            "--theme.primaryColor", "#1e3c72"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: uv add streamlit")

if __name__ == "__main__":
    main()
