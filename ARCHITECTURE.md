# Architecture

The design in one line: **the model proposes, code decides, and every claim has an address.**
Stage-by-stage detail, with the results and every decision made while building, is in `plan.md`.

```
 any pack ──► S0 PROFILE ──► S1 LEXICON + TRUST ──► S2 HYPOTHESES + EXAMPLE AUDIT ─┐
 (json, csv,   classify files,  test every meaning    test every rule; flag       │ S1 re-runs once with
  md, txt,     quarantine       against usage;        suspect examples            │ suspect examples
  srt, audio)  injections,      measure trust    ◄────────────────────────────────┘ down-weighted
               grounded rules
                    │
                    ▼
 S3 PLAN ──► S4 TRANSLATE ──► S5 VERIFY ──► S6 DECIDE ──► S8 REPORT
 who speaks   model, one call    independent    computed          srt, jsonl, review queue,
 to whom,     per scene, from    checks, no     confidence,       final report, release
 risk order   an evidence        model          one question      recommendation
              package
                    ▲                                 │
                    └──── S7 EVENTS: diff the learned state, redo only the lines whose dependencies changed
```

## Planning

- **Before translating, every line is analysed** (S3): speaker and addressee, the addressee's status relative to the
  speaker (from the character list; inferred addressees are marked as assumptions), relationship state (estranged or
  reconciled, from scene notes), tense, negation, question type, command, names, numbers, humour, and which Nadi-9
  words exist for each English word, with their S1 status.
- **Risk score per line**: missing words, single-source words, unresolved conflicts, status clashes, disputed word
  choices, documented grammar gaps, undocumented constructions, humour, numbers, timing. Scenes are processed
  **riskiest first**. After each correction the remaining scenes are **re-ordered** by the new risks.
- **Budget plan**: one model call per scene for translation; the rest of the 25/50 budget is reserved for replanning.
  Optional steps are the first to go if the budget runs short.

## Memory

- **Canonical pack** (S0): every file in the pack, whatever its format, becomes typed records (`LexEntry`,
  `Example`, `TextSection`, `Note`, `Segment`, `Character`, `EpisodeLine`, `RawEvent`), each with
  `ref = "<source>:<native id>"`, such as `dictionary_b:B-032`, `examples:E03`, `grammar:respect-2`.
- **Learned state**: word claims (`LexClaim`: status, chosen entry, every rival meaning with its evidence), tested
  hypotheses (`Hypothesis`: support and counterexample refs, per-tense evidence, competitors, conflict of interest,
  **version and history**), the example audit, and measured trust per source.
- **Decisions store their dependencies**: `lex:<word>`, `rule:<id>@<tense>`, `gap:<id>`, `social:<a>-><b>`.
  This is what makes targeted replanning possible.
- **Versioned run state**: every event increments the state version, writes a snapshot (`run_state/vNN.json`),
  and every decision records the version it was made at, so "what did the agent know when it decided S022?"
  has an exact answer.

## Verification

- **Independent by construction.** S5 uses no model and ignores the translator's account of its work. It
  re-analyses every word with the morphology learned in S0, looks each up in the S1 verdicts, re-runs the S2 rule
  checkers on the output (the same code that tested the rules against the examples), and builds its own back-gloss
  from the lexicon. A proposal that claims the right rule but uses the wrong pronoun is still caught (tested).
- **Seven checks**: vocabulary status, invented words and gaps, grammar (with *critical* for a word that is offensive
  to this addressee), names and numbers, omissions and additions, reading speed and form, and evidence citations
  (including citations of poisoned entries).
- **Confidence is computed, never asked of a model**: weakest word × weakest rule used × undocumented assumptions ×
  check results, and the weakest factor is named in `confidence_reason`. APPROVE needs ≥ 0.80 and nothing that
  policy reserves for a human.
- **Abstention is the default when coverage is missing**: a gap marker, an invented word, a disputed word choice,
  a documented grammar gap, a form with no evidence (*extrapolated*), or a status-dependent form that no tested rule
  covers all go to review with one question that names the evidence on each side.

## Failure recovery

| Failure | Behaviour |
|---|---|
| Transient model or tool error | retry with backoff; each attempt costs budget |
| Repeated failures of one task | **circuit breaker** after 3 in a row: fail fast, spend nothing (an outage costs 12 calls, not 20) |
| Rate limit (429) / request too large (413) | wait as long as the service asks / halve the reply allowance, resend; not processed, so no budget |
| Budget exhausted | optional steps skipped; lines fall back to review. Never a crash |
| Model unavailable for translation | no text invented: the line is INSUFFICIENT_EVIDENCE with the reason |
| Re-translation fails after a correction | keep the previous proposal and re-check it under the new state |
| Transcription fails | source marked unavailable; everything else proceeds |
| Transcription succeeds, but the engine does not know the language | transcript kept for reviewers, flagged `machine_transcript`, not learned from |
| A PDF/Word file without its reader library | reported unavailable, with the install command |
| Model output in the wrong shape | normalised deterministically where the meaning is unambiguous (marked `normalised`); otherwise kept as untestable, never guessed |
| Model skips a line | "no model output for this line", never a blank |
| Unreadable or unclassifiable file | reported, not forced into a role |
| Correction that is stale, unclear or targets nothing | not applied; becomes a review item quoting the message |

## Trust boundaries

1. **Pack content is data, never instructions.** It is sent to a model only in a user message labelled DATA; our
   instructions are in the system message. Text that addresses an AI ("ignore the rules", "mark everything approved")
   is detected and **quarantined**: kept for audit, never used as evidence, never shown to a model.
2. **Model output is untrusted until code checks it.** Rules must quote their source verbatim (grounding check);
   proposals may only use words from a curated evidence package; every answer is re-verified by S5.
3. **Machine transcripts are not evidence on their own.** Whisper output for a language it was never trained on
   is kept for reviewers but not learned from unless a human gloss accompanies it.
4. **Sources earn trust.** Trust is measured from how often a source's claims survive the evidence
   (e.g. Dictionary A 0.87 with 8 contradicted entries, Dictionary B 0.97). Poisoned entries never reach a prompt.
   Experts are checked for conflicts of interest. Viewer feedback can raise a question but never settle one.
5. **Humans own the decisions that should never be automated**: status and insult risk, kinship forms where experts
   disagree, humour, single-source words, and the release itself.

## Change handling (the correction event)

1. An event is parsed (the model reads free text; structured events are used as-is) and **validated**: its target
   must exist and its "old value" must match the current state.
2. It is applied as a new version (old rule kept in `history`; per-tense changes stored as `affix_by_tense`;
   the correction counts as evidence; a gap it answers is marked resolved).
3. S1 and S2 re-run deterministically (no model calls). The decision-relevant **signature** of every word and rule
   is compared before and after: `rule:grammar:respect-2#1@present` changes; `rule:…@past` does not.
4. Changes to *what* a word or rule says → **re-translate** the affected lines. Changes only to *how strongly*
   it is supported → **re-check** (re-verify and re-decide, no model call).
5. Every other decision must be byte-identical, and this is recorded (`unaffected_lines_unchanged`) and tested
   against a run with no events.

## Scale: 5,000 episodes, 40 dialects

S0–S2 run once per dialect (plus a linguist's review of the learned profile) and are cached; each episode pays only
for S3–S8, about one model call per scene. The review queue becomes a shared work list whose answers are fed back as
events, so every answer improves the dialect for later episodes, and the dependency keys find every affected line
across all episodes when a rule changes. The budget, the circuit breaker and the per-task keys are already per run.
