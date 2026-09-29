# Known limitations

What this submission deliberately leaves incomplete, and what would change with more time.

## Data
- **The official pack was not supplied.** Everything was built and measured on a synthetic pack I wrote, plus a
  second pack in a different language and format. My pack, my answer key and my code share my assumptions; the
  real run on Groq exposed one such blind spot (see `AI_COLLABORATION.md`), and the official pack may expose more.
- **Audio transcripts are for reviewers, not for learning.** With `--provider groq`, audio is transcribed by Whisper
  (`whisper-large-v3`), given the pack's vocabulary as a spelling hint. Whisper was never trained on this language:
  on synthetic Nadi-9 speech it produced "me gado vela nesh me tal velan…" for "mi gedo vela. nesh mi tolv velen".
  So a machine transcript without a human gloss is flagged `machine_transcript` and not learned from. The mock
  transcriber (no key) never invents speech; audio then stays unavailable.
- **PDF and Word files need `pypdf` / `python-docx`.** Without them such files are reported as unavailable, with the
  install command, rather than read.

## Learning
- **Source language is assumed to be English.** Gloss matching, tense cues and the synonym table live in
  `english.py`, which would need replacing for another source language.
- **The synonym table is small and hand-written** (e.g. polite = respectful = honorific).
- **Rule templates are a fixed set** (word order, affix, particle position, agreement, lexical choice,
  code-switching, gap).
  Rules outside them are kept as *untestable*: visible to reviewers, but not enforced.
- **Model extraction can be incomplete.** In one real run the model missed the polite-command rule. A safety net
  sends commands to higher-status addressees to review when no tested rule covers them; other kinds of missed rules
  have no such net yet.
- **Relationship and status inference is keyword-based** (character descriptions, scene notes). Every inference is
  shown as an assumption, but a subtle relationship could be misread.
- **Some evidence is not machine-readable.** For example, the interview anecdote showing `sorai` is insulting
  upward is not parsed; that dispute is left to a human.
- **Measured patterns are circular by design**: they are counted from the same data they are tested on, so they
  are used only where no stated rule exists, and never above warning severity. This includes the code-switching rule
  (English words kept unchanged, with known Nadi-9 endings), which only checks form, not whether a loanword is
  appropriate in context.

## Translation and verification
- **No LLM critic (V8).** All verification is code-based. A second-model critic, and the agreement-rate metric
  described in `plan.md`, were not built.
- **A skipped line is not retried.** If the model omits a line from a scene's answer, the line is marked
  "no model output" instead of being asked for again.
- **Spelling repair covers only morpheme boundaries** (e.g. `jan-a` → `jana`, learned from the examples).
  Verb order and other errors are reported, never rewritten.
- **Humour is always sent to review**; there is no automatic judgement of whether a joke survives.

## Decisions and events
- **Confidence weights and the 0.80 threshold are hand-set** and were checked only against the synthetic answer key
  (24/25 decisions agree, on both hand-written and real model output). They are not calibrated on real data.
- **Supported corrections:** per-tense changes to agreement rules, and corrections to approved examples.
  Other kinds (a dictionary meaning, a word-order rule) are parsed but not applied; they become review items.
- **Free-text corrections need a model** to be read. Without one, they become "please apply by hand".

## Engineering
- **Replay fixtures in `fixtures/replay/` are hand-written** stand-ins for model output, with planted mistakes.
  `fixtures/recorded/` and `fixtures/recorded_full/` are real `openai/gpt-oss-120b` output recorded on Groq.
- **Groq's free tier (8,000 tokens per minute) makes a real run take about 5–6 minutes**, mostly waiting.
- **Data policy:** I have not verified Groq's retention and training terms for this account; check them before
  sending the official pack (the brief forbids services that retain or train on it).
- **One episode per run; no user interface**; the SRT marks unfinished lines in the text itself, so
  `subtitles_approved.srt` is the file to put in a player.
