"""Test package. Run from the repo root:  python -m unittest discover -s tests -t . -v

The bot's modules import each other as top-level modules (`from storage import ...`),
exactly as they do when GitHub Actions runs `python src/main.py`, so src/ goes on the path here.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
