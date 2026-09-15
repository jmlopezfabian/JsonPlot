"""The evaluation harness: natural-language requests in, contracts scored.

Every model answer is recorded before it is scored, and the recording is
committed. Scoring is therefore a pure function of the recording and the code
under test, which is what lets `uv run evals` reproduce a published number on a
machine that has never run the model.
"""

from pathlib import Path

#: `evals/` — where the dataset and the recorded runs live, next to the package.
ROOT = Path(__file__).resolve().parents[2]
