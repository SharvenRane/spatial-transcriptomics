import os
import sys

# Make the repository root importable so ``import src`` works regardless of
# where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
