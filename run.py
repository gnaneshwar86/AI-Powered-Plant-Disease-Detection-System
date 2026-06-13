"""
Entry point for the LeafAI Plant Disease Detection application.
Run from the project root:
    python run.py
"""
import sys
import os

# Fix Windows cp1252 terminal encoding issues
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure the project root is on the Python path so 'src.*' imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app.main import create_app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"\nLeafAI Diagnostics running -> http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
