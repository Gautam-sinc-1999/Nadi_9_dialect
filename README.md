# Nadi-9 subtitle agent

An evidence-grounded agent that learns a language it has never seen from a small, contradictory and partly
poisoned pack, then proposes subtitles, and says clearly which ones it cannot vouch for.

The LLM only **proposes** (rules from prose, translations). Code **decides**: which sources to trust, which rules
hold, whether a subtitle is right, how confident to be, and what to ask a human. Every claim carries a reference
to the exact record in the pack it came from.

> **Note on data.** The official assignment pack was not supplied, so this repository includes a synthetic pack
> that reproduces every difficulty in the brief (`packs/synthetic/`), and a second, deliberately different pack
> (`packs/variant_veshti/`: another language, other file formats, no manifest) to show the system adapts rather
> than being fitted to one pack. Point `--pack` at the official pack to run on it.

## Quick start (no API key)

Requires Python ≥ 3.10 and `pydantic` ≥ 2.5 (`pytest` for the tests). If the pack has **PDF or Word files**
(a grammar note is often a PDF), also install `pypdf` and `python-docx`; without them those files are reported as
unavailable (the report says which, and why) instead of being read.

```bash
pip install "pydantic>=2.5" pytest pypdf python-docx

# the full run: learn the pack, plan, translate, verify, decide, absorb mid-run corrections, write the report
PYTHONPATH=src python3 -m nadi9 run --pack packs/synthetic --out runs/demo

# the same, replaying a recorded run of a real model (openai/gpt-oss-120b on Groq); this regenerates sample_run/
PYTHONPATH=src python3 -m nadi9 run --pack packs/synthetic --out runs/demo_real \
    --provider replay --fixtures fixtures/recorded_full

python3 -m pytest -q            # 220 tests, about 30 s, no network (UI tests skip without streamlit)
```

Open `runs/demo/final_report.md` first.

## Operate it in the browser (Streamlit)

```bash
pip install streamlit           # pandas comes with it
streamlit run ui/app.py         # opens http://localhost:8501
```

| Screen | What you do |
|---|---|
| **New run** | pick a pack in the project or **upload one** (files or a .zip), pick the model (recorded answers, Groq with a key from `.env` or typed in for the session, or no model), set the budget, start, and watch the progress |
| **Review** | go through the lines (filter by open / approved / rejected / needs re-review, by priority, or search); for each: the proposal, the agent's question, the evidence and checks; then **approve, edit and approve, or reject** |
| **Corrections** | submit a linguist's correction (a rule's ending, an approved example, or a free-text message read by the model). It is validated, applied, and only affected lines are redone; your reviews of changed lines are flagged for re-review |
| **Export** | the final subtitles (your decisions merged with the agent's), a working file with open lines marked, decisions with reviews, the review log and the agent's report |

Everything the UI does goes through `src/nadi9/operate.py`, the same code the tests use. A run folder is
self-contained (`run_config.json`, recorded model answers, `corrections.json`, `human_reviews.json`), so a run can be
reopened later, including after moving the project, without new model calls.

## With a real model (Groq)

```bash
cp .env.example .env            # add GROQ_API_KEY; .env is git-ignored
PYTHONPATH=src python3 -m nadi9 check-provider          # lists the models your key can use; sends no pack data
PYTHONPATH=src python3 -m nadi9 run --pack packs/synthetic --out runs/live \
    --provider groq --record fixtures/my_run            # --record saves every answer for keyless replay
```

`GROQ_MODEL` defaults to `openai/gpt-oss-120b`; `GROQ_MAX_TOKENS` defaults to 6000. The provider waits out rate limits
and shrinks oversized requests without spending budget. With `--provider groq`, **audio interviews are transcribed by
Whisper on Groq** (`GROQ_ASR_MODEL`, default `whisper-large-v3`), given the pack's own vocabulary as a spelling hint.
Whisper was never trained on this language, so a machine transcript without a human gloss is kept for reviewers and
not used as learning evidence.

### Running on a new pack (for example the official one)

| Mode | What you get |
|---|---|
| `--provider mock` or `replay` (no key) | everything that needs no model: the pack is read and profiled, meanings checked, sources scored, examples audited, every line analysed and risk-ranked. **No translations**: the mock never invents text, so every line comes back as "insufficient evidence" with its analysis and questions |
| `--provider groq` (key in `.env`) | the full run, including translations and audio transcription. Add `--record DIR` to keep the answers so the run can be replayed without a key | **Before sending the official pack to any hosted model,
check that service's data-retention and training terms** (the brief forbids services that retain or train on it).
Claude or another provider can be added behind the same `LLMProvider` interface (`src/nadi9/providers/base.py`).

## Commands

Each command runs every stage up to and including its own, and writes that stage's outputs.

| Command | Stages | Main output |
|---|---|---|
| `profile` | S0 read any pack, classify files, quarantine injected instructions, extract grounded rules | `pack_profile.json`, `canonical_pack.json` |
| `lexicon` | + S1 check every dictionary meaning against usage, measure source trust | `learned_lexicon.json`, `source_trust.json` |
| `hypotheses` | + S2 test every rule, audit the approved examples | `learned_rules.json`, `example_audit.json` |
| `plan` | + S3 who speaks to whom, what each line needs, risk order | `plan.json` |
| `translate` | + S4 model proposals per scene, structurally checked | `proposals.json` |
| `verify` | + S5 independent checks (no model) | `verification.json` |
| `decide` | + S6 computed confidence, decision, review question | `subtitle_decisions.jsonl`, `review_queue.json` |
| `run` | + S7 mid-run events and targeted replanning, S8 reports | everything below |
| `check-provider` | verify the Groq key and model | — |

Common options: `--provider {mock,replay,groq}` (default `replay`), `--fixtures DIR`, `--record DIR`,
`--max-model-calls 25`, `--max-tool-calls 50`. Failure drills: `--fail-model N`, `--fail-asr N`, `--hallucinate`.

## What a run produces (`sample_run/` is one)

| File | Contents |
|---|---|
| `subtitles.srt` | every proposal in episode order; unfinished lines visibly marked `[REVIEW Sxxx]` |
| `subtitles_approved.srt` | approved lines only |
| `subtitle_decisions.jsonl` | one record per line, in the brief's format plus audit fields |
| `review_queue.json` | open items, critical first, each with one precise question and its evidence |
| `final_report.md` | release recommendation, the brief's surprise events and how each was handled, trust, open gaps |
| `learned_rules.json` | every hypothesis with support, counterexamples, status, version history |
| `replan_log.json`, `run_state/` | the impact of every correction; what the agent knew at each state version |

`sample_run/` replays a real Groq run (9 of 25 model calls): 5 lines approved, 19 for review, 1 insufficient evidence,
**RELEASE AFTER REVIEW**. When the linguist's correction arrived mid-run, only S022 was redone, and the model itself
produced the corrected form `khanasi`, which passed every check. The hand-written replay (`--provider replay`
without `--fixtures`) plants model mistakes instead and ends **DO NOT RELEASE**, because one line calls the village
headman by a word that is insulting upward.

## Repository map

```
src/nadi9/
  profiler/      S0: discover, classify, mapping, rules (grounding), ruleshape (format + normaliser), morphology
  lexicon.py     S1   hypotheses.py S2   social.py   planner.py S3   translator.py S4
  verifier.py    S5   decider.py S6      run.py S7   report.py S8    pipeline.py (S0-S2)
  providers/     base (retry, circuit breaker), mock (replay), groq
  tools/         transcribe: mock (never invents) and Groq Whisper
  sanitize.py    injection detection     budget.py   english.py (source-language normaliser)   cli.py
packs/           synthetic/ (Nadi-9), variant_veshti/ (second pack), synthetic_answer_key.json (tests only)
fixtures/        replay/ (hand-written, with planted mistakes), recorded/ and recorded_full/ (real Groq output),
                 recorded_run1_before_schema_fix/ (kept as evidence, see AI_COLLABORATION.md)
tests/           220 tests: one file per stage, the Groq provider, audio, operations, the UI and its mapping to the backend
ui/app.py        Streamlit operator UI (src/nadi9/operate.py underneath)
sample_run/      the required sample run
plan.md          the design, stage by stage, with results and every decision made while building
```

## Where the brief's requirements are met

| Requirement | Where |
|---|---|
| Runnable CLI, setup instructions | this file |
| Structured run state, evidence references, decisions | `run_state/`, `subtitle_decisions.jsonl` (every `evidence` ref resolves to a pack record: tested) |
| Provider abstraction / mock mode | `providers/`; `mock` and `replay` need no key |
| Tests: conflict · unsupported term · correction event · tool failure | `test_lexicon.py` · `test_lexicon.py`, `test_verifier.py` · `test_run.py` · `test_profiler.py`, `test_translator.py`, `test_groq.py` |
| Sample run | `sample_run/` |
| Architecture note | `ARCHITECTURE.md` |
| AI collaboration log | `AI_COLLABORATION.md` |
| Known limitations | `KNOWN_LIMITATIONS.md` |
| Budget: 25 model / 50 tool calls, configurable | `budget.py`; one budget per run, enforced in code |
