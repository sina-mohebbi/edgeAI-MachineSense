"""Make the ml/ modules importable during tests.

pytest collects tests from ml/tests/ but the modules under test (config, model,
export_tflite) live in ml/. This puts ml/ on sys.path so `import config` works both
locally and in CI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
