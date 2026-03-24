#!/usr/bin/env python3
"""Entry point for running the backtest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from backtest_engine import main

if __name__ == "__main__":
    main()
