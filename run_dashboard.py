#!/usr/bin/env python3
"""
Simple launcher script for the Alpha Seeker Dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard."""
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected.")
        print("Please activate your virtual environment first:")
        print("source .venv/bin/activate")
        print()
    
    print("🚀 Launching Alpha Seeker Dashboard...")
    print("📍 Dashboard will open in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "alpha_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--theme.base", "light",
            "--theme.primaryColor", "#1e3c72"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()
