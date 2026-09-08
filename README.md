# Python for AI Systems — Masterclass

A progressive, English-language notebook course for generalist engineering students with varied backgrounds. The course starts with reproducibility and core Python reasoning, then connects data structures and systems concepts to modern AI engineering.

## Course map

| #     | Notebook                     | Central question                                                       |
| ----- | ---------------------------- | ---------------------------------------------------------------------- |
| 00    | Reproducible environments    | What must be captured so an experiment can be rerun?                   |
| 01    | Python, types and registries | How do we make dynamic systems extensible without making them fragile? |
| 02    | NumPy memory                 | When does an array operation copy data?                                |
| 03    | Tries                        | How can decoding be restricted to legal prefixes?                      |
| 04    | Heaps and beam search        | How do we retain the best candidates efficiently?                      |
| 05    | Async pipelines              | How do we overlap I/O without overwhelming a service?                  |
| 06    | DAGs and autograd            | How does reverse-mode differentiation traverse a computation?          |
| 07    | Vector search                | How do latency and recall trade off in approximate retrieval?          |
| 08    | Safe agent runtime           | How do deterministic controls surround an uncertain model?             |
| Bonus | Regex for AI systems         | Where is regex useful, unsafe, or simply the wrong parser?             |
| Bonus | Testing AI systems code      | What must a project test suite do that a scattered `assert` does not?  |

## Teaching contract

- All explanatory prose, code and project briefs are in English.
- Every worked example is complete and executable.
- Only the final project cell in each notebook is intentionally incomplete; these cells are tagged `project` (and `raises-exception`), and each ends in `raise NotImplementedError`.
- Projects include acceptance criteria and a **Checks to run yourself** list rather than hidden solutions.
- Diagrams are original SVG files stored in `assets/`, so the notebooks render offline and on GitHub.
- Examples use synthetic or local data and require no API key.

## Shared code — `course_utils.py`

Every worked example in a notebook is self-contained: you never need `course_utils.py` to run a notebook top to bottom. The module exists so the **project cells** can build on earlier notebooks without copy-paste — for example the notebook 04 project needs the trie from notebook 03:

```python
from course_utils import TokenTrie, mask_logits, decode_step
```

Each object in the module is the same implementation developed and explained in the notebook named in its section header, with a docstring added. It depends only on `numpy`. Run `python course_utils.py` to execute a self-check of every implementation's invariants.

## Notebook conventions

- **Predict, then run, then explain.** Each notebook opens with this loop, and `Predict` prompts appear before the cells where a wrong guess is most informative.
- **Assertions are the specification.** Most code cells end in `assert` statements. If one fails after an edit, the edit broke a stated invariant — that is the intended feedback. The `Testing AI systems code` bonus notebook develops this from an inline `assert` into a full project test suite (table-driven cases, error paths, isolation, `pytest` translation).
- **Measurements are shown, not just described.** Notebooks include runnable demonstrations of the trade-offs they discuss: streaming-heap vs full-sort timing (04), recall vs p50/p95 latency and candidates-vs-probes (07), catastrophic backtracking timing (bonus), finite-difference gradient checks (06), and an end-to-end constrained decode step (03).

## Setup

Python 3.11 or newer is recommended (the notebooks use `X | Y` type unions, `dict[...]` generics and top-level `await`).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks in numerical order. The async notebooks use top-level `await`, which works in Jupyter.

## Teaching plan

For a mixed-background cohort, plan approximately **20–24 hours** for guided teaching across the core notebooks and the two bonus notebooks, including demonstrations, prediction questions and checks. Add **10–14 hours** for project workshops, implementation and testing. Notebook 00 should receive 90 minutes because environment problems can block every later exercise; notebooks 05–08 are the most demanding and work best with extra workshop time. Notebook 06 now works up to reverse-mode autodiff through a by-hand chain-rule pass, and notebook 07 steps through k-means and the `probes` trade-off before the benchmark — budget the extra ~20 minutes each. The `Testing AI systems code` bonus pairs naturally with the first project workshop.

The [instructor transcript](INSTRUCTOR_TRANSCRIPT.md) provides suggested spoken explanations, questions and closing messages for every notebook. It is a teaching aid rather than a script that must be read word for word.

## Capstone project

Notebook 08 develops a small safe-agent runtime before leaving the final research-agent implementation open. The project should search a local document collection, retrieve passages, synthesize an answer with passage IDs, and abstain when the evidence is insufficient.

Expected project safeguards include:

- strict input and argument schemas with allowlisted, read-only tools;
- deterministic validation and authorization around model-generated proposals;
- retained retrieval scores and an evidence mapping for every factual claim;
- request deduplication, bounded timeouts and a structured audit log; and
- adversarial tests covering malformed input, unknown tools, bad arguments, timeouts, duplicates, denied side effects and cancellation.

Do not use `eval` or `exec`. Include a short risk register describing remaining limitations and assumptions.

## Assessment suggestion

Use prediction questions and modifications of worked examples formatively. Grade only the project artifacts, tests, explanation of trade-offs, and short technical report. A good submission should be correct, measured, readable and explicit about limitations. The `Testing AI systems code` bonus notebook defines what a passing test suite looks like — headless, reports every failure, covers boundaries and error paths, order-independent — and its project asks students to write one for an earlier notebook's project.

## Repository hygiene

Do not commit virtual environments, caches, generated secrets or large model files. Clear notebook outputs before a release if they expose machine-specific paths or noisy benchmark results.
