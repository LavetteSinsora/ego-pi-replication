"""Verification dashboard: replay the converted dataset (or the work-dir
stage outputs) through the deployment-shaped closed loop and render
everything a human needs to judge "is this data faithful and deployable?".

Entry points (run from the repo root with the repo venv):

    .venv/bin/python -m data_extraction.dashboard episode_1 -o report.html
    .venv/bin/python -m data_extraction.dashboard --batch --limit 3
    .venv/bin/mjpython -m data_extraction.dashboard.viewer episode_1
"""
