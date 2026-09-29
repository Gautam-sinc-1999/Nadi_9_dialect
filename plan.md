# Nadi-9 Subtitle Agent — Plan

Status: ✅ built and tested · 🔨 next · ⬜ planned

| Stage | What it does | Status |
|---|---|---|
| S0 Pack profiler | Read any pack, work out what each file is, normalise it | ✅ |
| S1 Lexicon and trust | Check every dictionary meaning against usage, score sources | ✅ |
| S2 Hypotheses and example audit | Test grammar rules, find wrong examples | ✅ |
| S3 Episode analysis and plan | Who talks to whom, what each line needs, risk order | ✅ |
| S4 Translate | LLM proposes Nadi-9 per scene, with evidence | ✅ |
| S5 Verify | Independent checks that can disagree with S4 | ✅ |
| S6 Decide and escalate | Confidence, decision, precise review question | ✅ |
| S7 Events and replanning | Apply corrections mid-run, redo only affected lines | ✅ |
| S8 Outputs and report | SRT, JSONL, review queue, release recommendation | ✅ |

---

## 0. Plan check: what I reviewed and changed

I checked every requirement in the brief against the design after building S0 and S1.

### Requirement coverage

| Brief requirement | Where it is handled |
|---|---|
| 1. Inspect evidence: sources, dates, authors, scope, reliability | S0 (authors, dates, roles) + S1 (measured trust) |
| 2. Create hypotheses: order, tense, negation, respect, relationships, code-switching | S0 extracts grammar rules; S2 adds data-induced, expert and usage-note hypotheses |
| 3. Test hypotheses, record counterexamples | S2 |
| 4. Translate with timing, relationships and tone | S3 (relationships, tone) + S4 |
| 5. Verify vocabulary, grammar, names, numbers, omissions, additions, reading speed, timing | S5 (checks V1–V8) |
| 6. Escalate with reason and precise question | S6 |
| 7. Replan: update only affected rules and lines | S7 (dependency index) |
| Per-subtitle JSON (translation or "insufficient", evidence, confidence + reason, assumptions, conflicts, decision, question) | S6 decision record |
| Budget: 25 model / 50 tool calls, configurable | `budget.py`, one budget for the whole run |
| Mock or replay mode | `providers/mock.py` + `fixtures/replay/` |
| Privacy: no retention or training | Mock/replay by default; the real provider (Groq) runs only with `--provider groq`. **Check the provider's retention/training terms before sending the official pack.** |
| Tests: conflict, unsupported term, correction event, tool failure | conflict ✅ unsupported ✅ tool failure ✅ correction ✅ |
| sample_run/: subtitles.srt, subtitle_decisions.jsonl, learned_rules.json, review_queue.json, final_report.md | S8 ✅ (real Groq run, replayable) |
| README, ARCHITECTURE, AI_COLLABORATION, KNOWN_LIMITATIONS | ✅ |

### Gaps found, and what changes

1. **The budget was split.** S0 had its own 10/20 budget. The brief says 25/50 *per episode*. → **One `Budget` object for the whole run.** Pack profiling is cached, so a second episode on the same pack does not pay for it again.
2. **Expert disagreements were not testable.** "dovanor only when estranged" (Varkey) vs "dovanor always on screen" (Rehn) are rules, not dictionary words. → **S2 extracts expert claims** into the same rule templates, with the same quote check, and tests them.
3. **Usage warnings inside glosses were ignored.** B-028 says `sorai` is "ONLY between equals". S1 kept that text but nothing enforces it. → **S2 turns usage notes into `lexical_choice` rules** (e.g. `sorai` requires addressee status = equal).
4. **Status and relationships were never computed.** Respect rules need to know "is the addressee higher than the speaker?". → **S3 builds a social graph** from the character list and scene notes. Every inferred relation is stored as an explicit assumption.
5. **S1 trusts all examples equally, including the two wrong ones.** → **S2 audits the examples**, then **S1 runs again** with suspect examples down-weighted (one feedback pass).
6. **Mid-run events need a run loop.** Events say "arrive after S012". → **S7 adds an orchestrator** that processes lines in plan order and fires each event when its trigger is reached.
7. **There was no safe fallback for translation.** If the model is down or the budget is spent, nothing must be invented. → The line gets **no translation**, and the decision is INSUFFICIENT_EVIDENCE with the reason "model unavailable".

---

# PART A — High level (the interview version)

## A1. The pitch in one paragraph

> Nadi-9 is a language no model has seen, and the material about it is incomplete, contradictory and partly poisoned. So I did not build a translator. I built a careful junior linguist. It reads the pack, **measures** how trustworthy each source is by checking it against real usage, turns the grammar into **testable hypotheses** and tests them, and only then lets an LLM **propose** subtitles. Every proposal is checked by code that is independent of the LLM, scored with a confidence **computed from the evidence**, and either approved or sent to a human with **one precise question**. Every claim carries a reference to where it came from. When a linguist sends a correction, a dependency index tells the system exactly which subtitles used that rule, and **only those are redone**.

## A2. The pipeline

```
            ┌──────────────── any pack (json, csv, md, txt, srt, pdf, audio) ───────────────┐
            ▼                                                                               │
 S0 PROFILE ─► S1 LEXICON + TRUST ─► S2 HYPOTHESES + EXAMPLE AUDIT ─┐                       │
 read, classify   check meanings       test rules, flag wrong        │ (S1 re-run once       │
 quarantine       vs usage, score      examples                      │  with suspect         │
 injections       sources ◄──────────────────────────────────────────┘  examples down-weighted)
            │
            ▼
 S3 EPISODE ANALYSIS + PLAN ─► S4 TRANSLATE ─► S5 VERIFY ─► S6 DECIDE + ESCALATE ─► S8 OUTPUTS
 who is talking to whom,       LLM proposes,    independent   computed confidence,     srt, jsonl,
 risk per line, order          per scene        checks        approve / review /       review queue,
                                                              insufficient             report
                                  ▲                                  │
                                  │      S7 EVENTS + REPLAN          │
                                  └──── correction arrives ──► dependency index ──► redo ONLY
                                                                     affected lines

 Everywhere: one run budget · provider = mock | replay | real · retry → fallback · versioned audit state
```

## A3. Five design principles

1. **The LLM proposes, code decides.** The model writes translations and turns prose into rule shapes. It never decides trust, confidence or approval.
2. **Every claim has an address.** `dictionary_b:B-032`, `examples:E03`, `grammar:respect-2`. Anything the system says can be looked up by a human.
3. **Trust is measured, not assumed.** A dictionary is trusted in proportion to how often its meanings survive contact with real usage. Size, date and a confident tone count for nothing.
4. **Unknown means ask, never guess.** No evidence → no Nadi-9 word. The line goes to review with a specific question.
5. **Changes are surgical.** Each decision records what it depended on. A correction touches only the decisions that depended on the corrected item.

## A4. How each surprise event is handled

| Surprise from the brief | What the system does | Stage |
|---|---|---|
| Poisoned dictionary entries | Each meaning is tested against the examples. `brem`="yesterday" fails because E03 says "tomorrow". 8/8 caught, 0 false alarms | S1 ✅ |
| Two approved examples are wrong | Examples that break several well-supported rules are flagged *suspect* and down-weighted before any correction arrives | S2 |
| A word changes meaning with status | The usage note on `sorai` becomes a rule. S3 knows the headman outranks Kabir. S5 raises a critical error if `sorai` is used upward | S2/S3/S5 |
| Linguist correction arrives mid-run | Parsed, validated, applied as a new rule version. The dependency index finds the affected lines; only those are redone | S7 |
| LLM confidently proposes an unsupported rule | Rejected twice: its quote is not in the source (S0), and it has zero supporting examples (S2) | S0 ✅ / S2 |
| Retrieved text says "ignore the rules" | Detected by pattern, quarantined, never used as evidence and never put into a prompt. Pack text is always passed as data | S0 ✅ |
| Model or transcription tool fails | Retry with backoff, then fall back: source marked unavailable / line marked "model unavailable". The run never crashes | S0 ✅ / S4 |

## A5. Answers to the interview questions

**How does your agent decide a source is trustworthy?**
It tests the source. Every dictionary meaning is checked against the approved examples and interviews. Trust = (prior × 4 + confirmed) / (4 + confirmed + contradicted). Dictionary A came out at 0.87 (61 confirmed, 8 contradicted) and Dictionary B at 0.97 (44 confirmed, 0 contradicted). Experts are checked for conflicts of interest: Rehn works for A's vendor, Kiran wrote B. Viewer feedback can raise a question but can never settle one.

**How would you detect that the verifier is repeating the translator's mistake?**
The main verifier is not an LLM. It is code that re-analyses every output word against the lexicon and re-applies the tested rules. It shares no prompt and no reasoning with the translator. If I add an LLM critic, it sees only the source, a word-by-word back-translation and the evidence table, never the translator's reasoning. I log how often the critic agrees with the translator on lines the code has flagged. A critic that always agrees is not independent.

**What changes when a linguist corrects one grammar rule?**
The rule gets a new version (the old one is kept, marked superseded). Every decision stores dependency keys such as `rule:grammar:respect-2@present`. The correction changes that key, the index returns the lines that used it (S022 only), and only those lines are re-translated, re-verified and re-decided. S021 uses `tavora`, not the present-tense honorific, so it is untouched. The replan log shows the before and after.

**Which decisions should never be automated?**
Status and insult-risk choices (`sorai` to the headman), kinship forms where experts disagree (`dovan` / `dovanor`), how humour is rendered, any line resting on a single unverified source, and the final release. The system prepares those decisions; a human makes them.

**How would this work for 5,000 episodes and 40 dialects?**
Profile and learn each dialect once (S0–S2), have a linguist review that profile, and cache it. Each episode then pays only for S3–S8. The review queue becomes a shared work list, and linguists' answers become new evidence for the dialect, so review load drops over time. The dependency index means one correction to a dialect finds every affected line across all episodes.

**What did an AI assistant suggest that you chose not to accept?**
Keep a running list in `AI_COLLABORATION.md`. Real examples from this build: a fuzzy column matcher that let `comment` be taken as an ID column; a character-window grammar lookup that missed "question … ne"; a substring check that counted "adverb" as "verb". All three were caught by the second, differently shaped pack and fixed with tests.

## A6. Budget plan (one run, defaults 25 model / 50 tool)

| Use | Model calls | Tool calls |
|---|---|---|
| S0 grammar rule extraction (+ classify / map only when heuristics fail) | 1 (+0–2) | – |
| S0 transcription of untranscribed audio | – | ≤ 3 per file (retries count) |
| S2 extract expert claims + usage-note rules | 1 | – |
| S2 model-proposed rules (optional) | 1 | – |
| S4 translation, one call per scene | 5 | 5 retrieval packages |
| S5 LLM critic, one call per scene (optional, first to be dropped) | 5 | 5 verification batches |
| S7 event parsing (free-text only) + re-translation of affected lines | ≤ 4 | ≤ 4 |
| **Typical total** | **~14–19** | **~15–20** |

"Tool call" means an external or expensive operation: transcription, a retrieval package for a scene, a verification batch, parsing an event. Budget checks happen in code before each call. When the budget runs out, optional steps are skipped first, then lines fall back to review.

---

# PART B — Technical, stage by stage

Every stage has the same shape: **input → steps → output**, the LLM use (if any), what happens on failure, and the tests.

## S0. Pack profiler ✅

**Goal:** accept any pack, in any reasonable format, and turn it into one standard shape.
**Code:** `src/nadi9/profiler/` · **Output:** `pack_profile.json`, `canonical_pack.json`

**Steps**
1. **Discover** (`discover.py`). Walk the folder and read each file by extension: JSON (finds the main record list, flattens nesting such as `experts[].notes[]` into rows with `parent_*` fields), CSV/TSV (delimiter sniffed), SRT/VTT (cues, plus "NAME: text" speaker prefixes), Markdown/TXT (split into sections by heading; a `[rule-id]` in a heading becomes the section ID; without headings, paragraphs become sections), PDF/DOCX (optional libraries), audio. A broken file is recorded as unavailable and the run continues.
2. **Classify roles** (`classify.py`). In order: (a) the pack's own manifest, if present; (b) structural heuristics that score each role from column names, record-list name, file name and content (e.g. `term`+`gloss` columns → dictionary; timecodes + speaker → episode; many grammar words → grammar; a column of multi-word non-English text next to an English column → examples); (c) one LLM call on a small sample (file name, keys, 3 rows); (d) otherwise **UNCLASSIFIED**, and the file is not used. A role needs a score ≥ 0.6 and a margin ≥ 0.15 over the runner-up.
3. **Map fields** (`mapping.py`). For each canonical field: pass 1, exact synonyms for all fields (so exact matches are never stolen); pass 2, fuzzy name match, required fields first; then **content inference** (the target-language column is found by its values: multi-word and almost no English function words); then an LLM proposal that is **accepted only if the proposed column exists**; otherwise the source is marked unavailable with the missing fields listed.
4. **Build canonical records.** Everything becomes `LexEntry`, `Example`, `TextSection`, `Note`, `Segment`, `Character`, `EpisodeLine` or `RawEvent`, each with `ref = "<source_id>:<native_id>"`.
5. **Interviews without a transcript** go to the transcription tool through `call_with_retry` (each attempt costs a tool call). On failure the source is marked **unavailable**. Segments with ASR confidence < 0.6 are flagged and not used for learning.
6. **Quarantine injections** (`sanitize.py`). Every text field is scanned with patterns such as "ignore … rules", "note to AI …", "mark all … approved", "confidence 1.0". Two independent hits → the record is flagged `quarantined:injection`, kept for audit, and excluded from evidence and prompts. (Found: A-060, VF-08.)
7. **Extract grammar rules** (`rules.py`). One LLM call per grammar file. The model must return rules in fixed templates (`word_order`, `affix`, `particle_position`, `agreement`, `lexical_choice`, `gap`, `untestable`), each with a **verbatim quote**. Code accepts a rule only if the template and parameters are valid and the quote is found in the cited section (after removing markdown and normalising spaces and dashes). Otherwise it is **rejected with a reason**. If the model is unavailable, each section is kept as `untestable`, so nothing is lost and nothing is invented.
8. **Measure the language's shape** (`morphology.py`). Roots come from the dictionaries; affixes from entries written `-x` / `x-`; **new affixes are induced** when an unknown ending or beginning attaches to known roots at least twice and is not just a chain of known affixes. Every example and good interview segment is analysed to count verb position (ignoring particles), particle positions (before/after verb, clause start/end) and unknown tokens (loanwords, names).

**LLM use:** grammar extraction (1 call); classification and mapping only when heuristics fail.
**Failure handling:** unreadable file → unavailable; no role → unclassified; model down → untestable rules; ASR down → source unavailable.
**Tests (16):** roles and counts; injections in both packs; hallucinated rule rejected; transcription failure contained; model outage fallback; budget = 0 still works; unknown file not forced; Pack 2 adaptation (verb-initial, prefixes, induced `zu-`); target column found by content; SRT speakers; exact-beats-fuzzy; "verb" ≠ "adverb"; quote grounding.

**Results.** Pack 1: 26 rules grounded, the invented rule rejected, verb-final 32/32. Pack 2 (different language and formats, no manifest): verb-initial 14/14, prefixes `ka-`, `zu-` (induced), `mo` after the verb.

## S1. Lexicon and source trust ✅

**Goal:** decide what each word means, based on evidence, and score each source by how often it was right.
**Code:** `src/nadi9/lexicon.py`, `src/nadi9/english.py` · **Output:** `learned_lexicon.json`, `source_trust.json` · **No LLM.**

**Steps**
1. **Word analyser.** `Morphology` built from the non-quarantined lexicon, with induced affixes. `analyse("velen") → vel + en`.
2. **Contexts.** Each example and each good interview segment becomes `Context(ref, roots, affixes, english_lemmas, weight)`. English is normalised by `english.py`: base forms ("went"→"go"), numbers ("ten"→"10"), register synonyms ("polite"/"honorific"→"respectful"). Weight: example 1.0; segment = its ASR confidence.
3. **Group entries by word:** `vel: [A-013 "come", B-010 "go"]`.
4. **Collect evidence per word:** `used_in` = contexts whose roots contain the word; `grammar` = every grammar **sentence** or table row that mentions it.
5. **Test each meaning** (`check_sense`). `parse_gloss` splits a gloss into core senses and usage-note words. A sense fits a context if at least half of its words appear in the context's English. Hits, misses, grammar hits and qualifier hits are recorded with refs.
6. **Judge** (`judge`). Group the entries whose meanings agree (share a core word), then:
   - **Disagreement** (`brem`, `vel`, `sorai`, …): rank groups by usage hits, then grammar hits. Winner fits usage and rival never does → RESOLVED_CONFLICT, rival CONTRADICTED. No usage, but grammar backs only one → resolved by grammar (`dem`). Anything else → UNRESOLVED_CONFLICT with **no meaning chosen**.
   - **Same core, clashing usage notes** (`tavo`: "plural" vs "respectful"): the grammar sentences decide which note is right, else unresolved.
   - **Agreement:** CORROBORATED if confirmed (the most specific supported gloss is chosen: "elder brother" over "brother"); AGREED_UNTESTED if several dictionaries agree but the word is never seen; SINGLE_SOURCE if one dictionary and never seen; UNSUPPORTED if used but no meaning fits.
   - Every verdict carries a `reason` naming its evidence.
7. **Discussion links.** Expert notes and feedback that mention the word are attached (`discussed_in`) but never counted as votes.
8. **Trust.** Per dictionary: `trust = (prior × 4 + confirmed) / (4 + confirmed + contradicted)`; untested entries count for nothing. Other sources keep their prior until a later stage tests them (marked `measured: false`).
9. **Conflicts of interest.** An expert's name inside a source's author field, or the author's organisation inside an expert's role.

**Tests (16):** every poisoned entry caught (checked against the answer key); no false alarms; resolved meanings; `sorai` keeps its warning and the dissent stays visible; A-only words are single-source; the quarantined entry never becomes a claim; B trusted above A; conflicts of interest on both sides; Pack 2 fully corroborated; an unresolved conflict makes no guess; unsupported meaning flagged; a grammar sentence can settle a conflict.

**Results.** 8/8 poisoned entries, 0 false alarms. A: 0.87, B: 0.97. 26 single-source words.

## S2. Hypotheses and example audit ✅

**Goal:** turn every rule-like claim into a tested hypothesis with support and counterexamples, and find examples that are probably wrong.
**Code:** `src/nadi9/hypotheses.py`, `src/nadi9/social.py`, `src/nadi9/pipeline.py` · **Output:** `learned_rules.json`, `example_audit.json`
**Run:** `PYTHONPATH=src python3 -m nadi9 hypotheses --pack packs/synthetic --out runs/s2 [--hallucinate]`

**Results on Pack 1** (3 of 25 model calls):
- Suspect examples: **exactly E09 and E14**, found before any correction event. E09: breaks `respect-1#2` (tu for equals/younger) + challenged by KIR-1. E14: breaks both `tense-1#2` (past) and `tense-1#3` (future).
- The model's invented rule ("questions put the verb first"): **refuted**, 0 support, 4 counterexamples.
- REH-1 ("always dovanor on screen"): **refuted** by the approved examples E07 and E18, with the conflict of interest shown. VAR-1 ("dovan after reconciliation"): supported. They are linked as competitors.
- `sorai`: REH-3 (safe upward) has no evidence either way and competes with KIR-4 and B-028. Left for a human.
- `respect-2`: past and future supported, **present has zero evidence**, marked extrapolated.
- Trust is now measured for examples (0.90, 2 suspect), grammar (0.95, 12 confirmed, 0 refuted) and expert notes (0.72, 1 refuted).
- Pack 2: rules measured purely from data (verb-first, `mo` after the verb, `ne` at the start), **no false suspects**.

**Details decided during the build**
- **Leave-one-out audit.** An example counts as breaking a rule only if the rule is SUPPORTED *without* that example. Otherwise a wrong example could protect itself by being the counterexample that weakens the rule.
- **Violations are counted per topic** (e.g. all rules about `tu` together), so two sources restating the same rule do not double the suspicion.
- **Endings whose meaning cannot be read off the English** (command, location, plural) are capped at WEAK: seeing `-sa` proves the ending exists, not that it marks commands.
- **Status `elder_sibling` is not judged by the tu/tavo rules,** because the grammar itself says address to elder siblings varies.

**Steps**
1. **Collect hypotheses from five places**, all in the same templates:
   - (a) grammar rules from S0 (grounded quotes);
   - (b) **data-induced** rules from the measured shape (e.g. verb-final 32/32, `na` before the verb);
   - (c) **expert claims**: one LLM call `extract_claims` over expert notes (e.g. VAR-1 "dovanor while estranged, dovan after reconciliation" → `lexical_choice(term=dovanor, condition=relationship.estranged)`), with the same quote check, each tagged with its expert and any conflict of interest;
   - (d) **usage notes from the lexicon**: glosses with restrictions ("ONLY between equals") → `lexical_choice(term=sorai, condition=addressee_status=equal)`, from the same call;
   - (e) optional **model-proposed** rules from the examples (1 call). They must earn support like everything else.
2. **One checker per template**, run on every example and good segment:
   - `word_order`: verb position of each clause.
   - `affix` with a feature: the tense cue in the English side (past: did, went, -ed, yesterday, ago, last; future: will, tomorrow; present: neither) must match the affix.
   - `particle_position`: is `na` directly before the verb?
   - `agreement`: if `tavo` is the subject, does the verb carry `-sa`? Counted **per tense**, so "present: 0 examples" is visible.
   - `lexical_choice`: the example's context (speaker, addressee, relationship) goes through `social.py` → addressee status higher / equal / lower → is the term allowed?
   - `gap` and `untestable`: not tested; stored as **known unknowns** that later flag any line needing them.
3. **Score each hypothesis:** `support`, `counter` (with refs), `confidence = (support + 1) / (support + counter + 2)`. Status: SUPPORTED (≥ 2 support, counters ≤ 20%), CONTESTED, UNSUPPORTED (0 support), REFUTED (counters > support). Feature coverage is kept, e.g. `respect-2`: past 3, future 1, **present 0 → extrapolated**.
4. **Competing hypotheses stay side by side.** VAR-1 vs REH-1 on `dovanor`: approved examples E07/E18 use `dovan` in close sibling scenes, which counts against "always dovanor". The result is shown as evidence for each side, never collapsed silently.
5. **Example audit.** For each example: `suspicion = Σ (confidence of each SUPPORTED rule it violates) + 0.5 × (expert notes that challenge it)`. Suspect if ≥ 0.8. Expected: **E09** (familiar `tu` to a boss violates respect-1, and KIR-1 challenges it) and **E14** (future `-iru` with an English past "left", violating tense-1).
6. **Feedback pass.** Suspect examples get weight 0.3, then S1 and S2 run once more. Stop after 2 passes or when nothing changes.

**LLM use:** `extract_claims` (1 call), `propose_rules` (optional, 1 call). **Failure:** model down → only grammar and induced hypotheses; expert notes stay as discussion.
**Tests:** model-proposed rule with 0 support → UNSUPPORTED; E09 and E14 flagged suspect *before* the events; `respect-2@present` marked extrapolated; VAR-1 and REH-1 both present with their evidence; a suspect example no longer counts as full evidence in S1.

## S3. Episode analysis and plan ✅

**Goal:** understand each line before translating it, and decide the order and budget.
**Code:** `src/nadi9/planner.py`, `src/nadi9/social.py` · **Output:** `plan.json` (lines, social graph, scene order, budget)
**Run:** `PYTHONPATH=src python3 -m nadi9 plan --pack packs/synthetic --out runs/s3` · **No LLM.**

**Results on Pack 1**
- Social graph: Riya→Arun elder sibling, Ama→Arun lower, Kabir→Arun equal, Kabir→Bhola higher (headman), Arun→Mr. Dass higher (boss). Each has a reason.
- **S012 is the top risk (13.5)**: `sorai` said to the headman, a status clash. So SC3 goes first.
- **S022 hits the present-tense honorific gap; S021 does not** ("your food" needs a possessive, not an honorific verb). This is exactly the split EV01 will test.
- Pronoun choice: `tavo` for the headman, `tu` for Ama→Arun and Bhola→Arun, "rules do not decide" for the elder brother.
- Gaps found before translation: `boy` (S009), `late` (S017); negative command (S017), future negation (S020), `ko` with a question word (S005), clock time (S015, not documented anywhere).
- Risk 0 for S004, S016, S021.
- Pack 2 (no addressee field): "Grandfather, …" → Oran via his description; other lines by turn-taking; line 1 has no inference, so it is marked unknown and its risk is raised.

**Details decided during the build**
- "Thank you" is a formula, not an address, so it triggers no pronoun choice.
- Pronoun alternatives (tu/tavo) are a **choice** ("rules point to tavo"), not a clash. A clash is only reported for the single word a line would use (`sorai` → headman).
- A gap topic that names several things ("future negation; negative commands") is split, and each part matches on its own.
- A gap that names a word ("when dovanor is used") applies only where that word is a candidate.
- The honorific gap needs "you" as the subject, stated or implied by a command.
- Loanwords are checked in their written form, so "meeting" is not stemmed to "meet" first.
- **Not built:** the optional LLM call for unclear relations. Heuristics plus visible assumptions were enough on both packs.

**Steps**
1. **Social graph** (`social.py`). For each speaker→addressee pair: relationship (sibling, parent→child, employee→boss, villager→headman, friends) and addressee status (higher / equal / lower), from character descriptions (kinship and role words, ages) and scene summaries. Each relation is stored as an **assumption** with its source: `"Arun is Riya's elder brother" ← characters:Arun (age 31 vs 24)`. Scene states come from scene summaries: SC1 *estranged*, SC5 *reconciled*. Unclear relations → one optional LLM call, whose output is still recorded as an assumption, not a fact.
2. **Line features** (source side, deterministic): tense, negation, yes/no or wh-question, command, vocative (a leading name or title + comma), numbers, names, humour (from direction notes like "joke", "teasing"), content words.
3. **Coverage pre-check.** A reverse index (English lemma → Nadi-9 terms whose chosen meaning contains it) finds a candidate for each content word, with its S1 status. No candidate → a **gap** (e.g. "late", "change"). A single-source candidate → a weak spot (e.g. "door" → `dovar`).
4. **Risk score per line:** +3 per missing word, +1.5 per single-source word, +3 per unresolved conflict, +2 status-sensitive choice (`tavo` / `sorai`), +2 kinship with a known gap, +3 a known grammar gap (future negation, negative command), +1.5 humour, +1 numbers, +1 timing already too tight.
5. **Plan:** scenes ordered by highest risk, one translation batch per scene, budget reserved per step, and the reason for the order written down. Re-planned after every event.

**LLM use:** optional, 1 call for relations the heuristics cannot settle.
**Tests:** Riya→Arun = elder sibling; Kabir→headman = higher; S012 gets top risk; S017 flagged for the negative-command gap before translation.

## S4. Translate ✅

**Goal:** get a Nadi-9 proposal for each line, with the evidence it claims to use.
**Code:** `src/nadi9/translator.py` · **Output:** `proposals.json`
**Run:** `PYTHONPATH=src python3 -m nadi9 translate --pack packs/synthetic --out runs/s4 [--fail-model 9]`

**Results on Pack 1** (8 model / 6 tool calls so far in the run)
- 24 proposals, and S017 honestly declined ("no word for 'late', negative commands undocumented").
- The replay answers contain planted model mistakes. Code already catches the structural ones: `hapo` **invented** (S010), a **non-existent rule** cited (S013), `manu` **outside the package**, a possible addition (S008). The rest (`sorai` to the headman, the dropped "first", the poisoned citation A-066 in S016, familiar commands to the headman) are left for Stage 5, which is where they belong.
- The package never contains quarantined text, contradicted entries or the refuted model rule (tested).
- Model outage: nothing is invented, and the **circuit breaker** stops after 3 failed attempts per task, so an outage costs 12 calls, not 20.
- Pack 2 has no replay answers: every line gets "no model output", never a made-up translation.

**Details decided during the build**
- The package also includes words for small English words ("now", "very", "is", "with") that Stage 3 does not count as gaps. Without them, correct words were wrongly flagged as invented.
- "Invented" (exists in no usable source) and "outside package" (a real word the model was not given) are different flags with different severity.
- A **circuit breaker** was added to `call_with_retry`, keyed per task, so one failing tool never blocks another.
- **Real provider: Groq** (`providers/groq.py`), OpenAI-compatible API through the standard library, so no new dependency. Key and model come from `.env` (`GROQ_API_KEY`, `GROQ_MODEL`; `.env` is git-ignored, and `.env.example` is committed). Our instructions go in the system message; pack content goes in a user message labelled DATA. Temperature 0, JSON mode. 429/5xx/invalid JSON → retryable (honours `Retry-After`); 401/403 → stop with "check GROQ_API_KEY". The key never appears in names or errors (tested).
- **Record and replay real runs:** `--provider groq --record fixtures/recorded` saves every real answer in the replay format; `--provider replay --fixtures fixtures/recorded` replays it without a key.
- `python -m nadi9 check-provider` lists the models the key can use and sends one tiny test request, with no pack data.
- Claude remains optional: one more file behind the same `LLMProvider` interface.

**Steps**
1. **Context package per scene (retrieval tool call).** Lines with features and social assumptions; **only** the relevant lexicon entries (reverse-index candidates + function words + affixes) with their status and refs; SUPPORTED and CONTESTED rules (never REFUTED); known gaps. Quarantined text is never included. Pack text sits in the data part of the prompt, not the instructions.
2. **One LLM call per scene.** The prompt contract: return, per line, `nadi_9_text` or `null`; `tokens` (form, root term, affixes, claim ref); `rules_used` (id + feature, e.g. `grammar:respect-2@past`); `assumptions`; `gaps` (English words with no evidence). A gap must be written as `[?word]`, and the model may use no term outside the package.
3. **Structural pre-check:** every token's root must be in the package. Anything else is marked `invented` for S5.
4. **Modes:** `replay` uses recorded or hand-authored responses for the synthetic episode (labelled as such); `mock` on an unknown pack produces no translation, so lines go to review; `anthropic` is the real provider, with a `--record` flag that saves responses as replay fixtures.
5. **Failure:** retry 2×; then every line in the scene gets `nadi_9_text = null`, reason "model unavailable".

**Tests:** an invented token is caught; a gap stays a gap; model outage → no text invented; the prompt contains no quarantined text.

## S5. Verify ✅

**Goal:** independent checks that can disagree with the translator.
**Code:** `src/nadi9/verifier.py` · **Output:** `verification.json` · **No LLM.**
**Run:** `PYTHONPATH=src python3 -m nadi9 verify --pack packs/synthetic --out runs/s5 [--fixtures fixtures/recorded]`

**Why it is independent:** it never uses the translator's own account of its work. It re-analyses every word
with the morphology learned in S0, looks each up in the S1 verdicts, re-runs the S2 rule checkers on the output
(the same code that tested the rules against the examples), and builds its own back-gloss from the lexicon.
Tested: a proposal that *claims* the right rule but uses `tu` for the headman is still caught.

| Check | What it looks at | Severity |
|---|---|---|
| V1 vocabulary | each word's S1 status; endings that exist in no source | single-source → warning; unresolved/unsupported → error |
| V2 invented | words in no usable source; `[?gap]` markers; S4 structural flags | error ("outside package" → warning) |
| V3 grammar | S2 rules (SUPPORTED/WEAK) re-applied; register words vs addressee; S3 pronoun choice; command form | stated rule → error; measured pattern → warning; a forbidden word for this addressee → **critical**; a form with no evidence → warning/info "extrapolated" |
| V4 names, numbers | names used in the English appear; numbers use the lexicon's number word | error |
| V5 meaning | back-gloss vs the S3 coverage: omissions and additions | warning (pronoun dropped → info) |
| V6 form, timing | duration, reading speed, length; hyphens the approved examples never write | warning + suggested fused spelling |
| V7 evidence | cited entries that are contradicted (poisoned) or do not exist; reliance on refuted rules | warning |

**Results.** Hand-written answers: every planted mistake caught (S012 critical, S008 omission + addition,
S016 poisoned citation, S010 invented, S013 fake rule, S011 timing). **Real Groq output:** every real error caught
(verb order in S001/S005/S008/S012/S019/S024, `na` placement S003, missing tavo/-sa S022 and S007, register
`tavora` S023, "friend" silently dropped S012, stray `?and` S024, hyphenated endings with fused spellings suggested).
**Dependencies are recorded per line:** S022 depends on `rule:grammar:respect-2#1@present`, S021 does not.

**Bugs found while building it**
- `ama` is both "mother" and the character Ama: a token is a name only if the English line uses that name.
- "has come back" is a present perfect: the English tense cues now add *past* for have + participle, and treat "now" as present.
- Measured rules duplicated stated ones and invented rules like "ko after the verb": measured rules now apply only
  where no stated rule covers the topic, and never above warning.
- The real model put "names"/"loanword" or two refs joined with `|` in citation fields: V7 now judges only
  things shaped like references.
- `tosh-guna` is one dictionary word containing a hyphen: a whole-word match now comes before splitting.
- **Not built:** V8, the optional LLM critic.

## S6. Decide and escalate ✅

**Goal:** a calibrated confidence with a reason, a decision, and one precise question.
**Code:** `src/nadi9/decider.py` · **Output:** `subtitle_decisions.jsonl`, `review_queue.json` · **No LLM.**
**Run:** `PYTHONPATH=src python3 -m nadi9 decide --pack packs/synthetic --out runs/s6 [--fixtures fixtures/recorded]`

**Results**
- Hand-written answers: **24/25 decisions match the answer key.** The one difference (S013) is correct:
  that answer cites a rule that does not exist, so it goes to review.
- Real Groq output: **24/25.** The difference (S013) is a real omission: the model wrote "go" for "go back".
- Every evidence reference resolves to a record in the pack (tested). Every review item has a question and a priority;
  every approval has confidence >= 0.80 and no question (tested).

**Example questions it writes**
- S012 (critical): *"Should Kabir's line to Bhola (higher status) keep 'sorai'? It is meant as cheeky or rude, or it should
  be respectful. Evidence: "friend. ONLY between equals." (dictionary_b:B-028; also expert_notes:REH-3, KIR-4, VF-01)."*
- S001: *"Which kinship/address form should Riya use to Arun (relationship: estranged)? … VAR-1#1 says 'dovanor' only when
  estranged [weak: 0.91 for / 0 against; conflict of interest: Varkey authored interview_01] vs REH-1#1 says 'dovanor'
  always [refuted: 0 for / 2 against (E07, E18); conflict of interest: Rehn works for the vendor of dictionary_a]."*
- S022: *"The sources say present-tense honorific is undocumented (VAR-2) … The proposal assumes 'hulmar, tavo khanasa ko?'.
  Is that correct?"* — exactly the question EV01 will answer.

**Details decided during the build**
- `agreed_untested` weighs 0.85 (not 0.75 as first planned): two dictionaries agreeing is real, if weak, evidence;
  at 0.75 "yes" (`ha`) could never be approved.
- Grammar uses the *weakest rule actually used* with fixed levels (1.0 / 0.9 / 0.75 / 0.6 extrapolated), not a product of
  Laplace confidences, which punished every line for being long.
- V7 (citation) problems are reported but do not lower confidence in the subtitle itself.
- Spelling fix: when V6 finds written-out morpheme boundaries, the fused spelling is **re-verified from scratch** and adopted
  only if it raises nothing new; the model's text stays in `original_text`. (S014, S016, S021 on the real output.)
- The verifier passes structured facts (`Issue.meta`: rule id, tense, term) so questions never depend on parsing message text.
- A line is INSUFFICIENT_EVIDENCE if it has no translation or at least half its content words are gaps; a single gap is a review.

**Confidence is computed, never asked of the model** (as built; see weights in `decider.py`).
```
lexical      = min over content tokens of status weight
               corroborated 1.0 · resolved 0.9 · agreed_untested 0.75 · single_source 0.55 · unresolved 0.3
grammar      = product of the confidences of the rules used (extrapolated feature → at most 0.6)
context      = 0.9 ^ (number of unverified assumptions that change a word form)
verification = 1.0, × 0.85 per warning, capped at 0.35 if any error
confidence   = lexical × grammar × context × verification
reason       = the weakest factor in words, e.g. "0.52: 'yoru' (year) is single-source (dictionary_a:A-042)"
```

**Decision**
- **INSUFFICIENT_EVIDENCE:** no translation, or an essential content word is a gap.
- **HUMAN_REVIEW:** confidence < 0.80, any error or critical issue, an unresolved conflict, a known grammar gap, or a humour line with a literal rendering.
- **APPROVE:** everything else.
- **Priority:** critical (insult or meaning reversal), high, normal.

**Review question:** built from the single most decisive cause, with the evidence named. Examples:
- *Status:* "Kabir greets the headman with 'sorai'. Among equals it means 'friend'; said to a superior it is insulting (B-028, KIR-4, interview_03). Is this meant as a cheeky joke the audience should hear as rude, or should it be respectful?"
- *Kinship:* "After reconciliation, does Riya say 'dovan' (VAR-1, interview_01 seg2) or 'dovanor' (REH-1)?"
- *Gap:* "Future negation is undocumented (grammar:neg-1). KIR-3 has heard two patterns, 'na veliru' and 'velnai'. Which applies to 'I will not leave' (nas)?"
- *Single source:* "Only Dictionary A lists 'yoru' (year). Is it correct?"

**Record** (the brief's JSON, extended):
```json
{
  "subtitle_id": "S012",
  "source_text": "Welcome, friend! Come, sit with us.",
  "nadi_9_text": "salo, sorai! ...",
  "confidence": 0.31,
  "confidence_reason": "critical: 'sorai' addressed to a higher-status person",
  "decision": "HUMAN_REVIEW",
  "priority": "critical",
  "evidence": ["dictionary_b:B-028", "examples:E10", "interview_03:seg1"],
  "assumptions": ["Bhola (headman) outranks Kabir (characters:Bhola)"],
  "conflicts": ["dictionary_a:A-053 vs dictionary_b:B-028 (resolved: B)", "expert:REH-3 dissents"],
  "review_question": "...",
  "checks": {"V1": "pass", "V3": "critical", "V6": "pass"},
  "depends_on": ["lex:sorai", "lex:salo", "rule:hypothesis:sorai-equals-only", "social:Kabir->Bhola"],
  "state_version": 7
}
```

**Tests:** confidence falls with a single-source word; an unresolved conflict never auto-approves; the question names the evidence; a line with no translation → INSUFFICIENT_EVIDENCE.

## S7. Events and replanning ✅

**Goal:** absorb corrections mid-run and redo only what they affect.
**Code:** `src/nadi9/run.py` · **Output:** `replan_log.json`, `run_state/vNN.json`, plus every earlier output
**Run:** `PYTHONPATH=src python3 -m nadi9 run --pack packs/synthetic --out runs/full [--fixtures fixtures/recorded]`

**How it works**
1. **Run loop:** scenes in risk order; translate → verify → decide; then fire events whose trigger line is in that scene.
   Events never reached fire at the end. After an event the remaining scenes are re-ordered by the new risks.
2. **Parse and validate:** structured events are used as-is; free text goes to the model (`parse_event`). A correction
   is applied only if its target exists and its "old value" matches the current state; otherwise it becomes a review item.
3. **Apply as a new version:** a rule keeps its history (`version`, `history`); a per-tense change is stored as
   `affix_by_tense`; the correction itself becomes evidence (`external_support`); a gap it answers is marked `resolved_by`;
   a corrected example keeps its old text in `history` and stops being down-weighted.
4. **Impact by diff:** S1/S2 re-run deterministically (no model calls). The signature of every word
   (status, meaning, entry) and rule (strength, shape, and per tense: attested? ending?) is compared before and after.
5. **Two kinds of change:**
   - *what* a word or rule says changed → the line is **re-translated** (one model call per scene group);
   - only *how strongly* it is supported changed → the line is **re-checked** (re-verify + re-decide, no model call).
   If a re-translation fails, the previous proposal is kept and re-checked under the new state.
6. **Proof:** every other decision must be byte-identical (`unaffected_lines_unchanged`), and a snapshot is written per version.

**Results on Pack 1** (9 model calls in total: 3 learning, 5 scenes, 1 replan)
| Event | Changed keys | Lines | Result |
|---|---|---|---|
| EV03 (E09 corrected) | the `tavo` rules got stronger | S022 re-checked | unchanged; the audit had already flagged E09 |
| EV01 (present ending `-sa` → `-si`) | `rule:grammar:respect-2#1@present`, `lex:-si`, `gap:VAR-2#1` | **S022 re-translated** | `khanasa` REVIEW 0.54 → **`khanasi` APPROVE 0.9** |
| EV02 (E14 corrected) | none | none | the audit had already flagged E14 |

- Compared with the same run **without** events: only S022 differs (tested against the answer key's `must_not_touch`).
- A correction that arrives *before* its line is reached just changes the state; the stale fixture answer `khanasa` is then
  caught by the corrected rule. A repeated or unknown correction is refused as stale. Pack 2's free-text email with no model
  available becomes "please apply by hand". On the real output (no recorded re-translation), S022 keeps the model's text,
  and the missing `-si` is now a hard error, because the present form is attested by the correction.

**Details decided during the build**
- EV01's trigger moved from S012 to S005. In risk order S012 is in the *first* scene, so the correction would have arrived
  before S022 was decided and nothing would need redoing. After S005 it arrives halfway (3 of 5 scenes). Both timings are tested.
- The first version re-translated S022 for EV03 just because the `tavo` rule got stronger, and the replay had no answer
  for that call. That led to the re-translate / re-check split.
- Decisions also depend on the gaps and disputed rules their line touches (`gap:<id>`, `rule:<id>` from Stage 3).

## S8. Outputs and report ✅

**Code:** `src/nadi9/report.py`, `nadi9 run` · **Sample:** `sample_run/` (a replay of a recorded real Groq run)
**Regenerate the sample, no key needed:**
`PYTHONPATH=src python3 -m nadi9 run --pack packs/synthetic --out sample_run --provider replay --fixtures fixtures/recorded_full`

| File | Contents |
|---|---|
| `subtitles.srt` | Every proposal in episode order; unfinished lines visibly marked `[REVIEW Sxxx]` / `[NO SUBTITLE Sxxx: …]` |
| `subtitles_approved.srt` | Approved lines only (safe subset) |
| `subtitle_decisions.jsonl` | One S6 record per line |
| `learned_rules.json` | S2 hypotheses (support, counterexamples, versions, history) + rejected claims |
| `review_queue.json` | Open items, critical first, each with question, evidence, back-gloss, suggested fix |
| `final_report.md` | Release recommendation, review queue, approved lines with reasons, the brief's surprises, corrections, trust, open gaps |
| `replan_log.json`, `run_state/vNN.json` | Every event's impact; what the agent knew at each version |
| also | `proposals.json`, `verification.json`, `plan.json`, `learned_lexicon.json`, `source_trust.json`, `example_audit.json` |

**Release rule:** open critical → DO NOT RELEASE; other open items or unapplied corrections → RELEASE AFTER REVIEW; otherwise READY.
Hand-written answers: DO NOT RELEASE (S012 calls the headman `sorai`). Real run: RELEASE AFTER REVIEW (the model avoided it).

**The real full run (Groq, gpt-oss-120b, 9 model calls, ~5.5 min), recorded in `fixtures/recorded_full/`:**
5 approved, 19 review, 1 insufficient. When EV01 arrived, **the real model re-translated S022 to `khanasi` itself**,
which passed every check and was approved; nothing else changed. The model avoided `sorai` for the headman by leaving
"friend" as a gap, and the review question says so.

**Found and fixed while building it**
- *Groq 413 "request too large":* the reply allowance counts against the per-minute token limit. The provider now halves it
  and resends (not processed, so no budget), and `GROQ_MAX_TOKENS` defaults to 6000.
- *An incomplete extraction:* the real model missed "Respectful command: root + -sa", so no tested rule covered polite
  commands and "ama mi ush" (a familiar command to his mother) was approved. **Safety net:** a command to a higher-status
  addressee with no tested rule for respectful commands goes to review ("cannot be checked for politeness").
- *A wrong question:* when the model left "friend" untranslated, the question claimed no word exists. Now it says
  "left untranslated although the lexicon has 'sorai' … probably because the addressee is higher. Use it, or rephrase?"
- *'?word' gap markers:* normalised to `[?word]` and noted, not read as invented words.
- *The model silently skipped a line* (S018): marked "no model output for this line", never left blank.
- *Non-determinism:* a set was iterated when building one line's conflict list, so its order changed between processes.
  Fixed; the suite now passes under several `PYTHONHASHSEED` values, and a test checks that `sample_run/` is exactly what
  the replay produces.

---

# PART B2 — First real model runs (Groq, openai/gpt-oss-120b)

Recorded answers: `fixtures/recorded/` (run 2) and `fixtures/recorded_run1_before_schema_fix/` (run 1).
Replay either without a key: `--provider replay --fixtures fixtures/recorded`.

**Run 1: every real rule was rejected.** The model understood the grammar and quoted it correctly, but wrote rules in
its own format (`{"order": "SOV"}`, one `tense_suffixes` table, `"allowed_status": "higher"`, `"term": "you (familiar)"`).
The validator rejected all 23 grammar rules, so E09/E14 were no longer caught. The system failed *safely*, but lost the
model's work. Cause: my prompts named the templates but never spelled out their fields, and my hand-written replay
answers matched my code only because I wrote both. **This is the "same author" blind spot, found by the first real run.**
Translation also failed on Groq's free-tier limit (8,000 tokens/minute), handled safely by the circuit breaker.

**Fixes**
1. `profiler/ruleshape.py`: one exact format spec (`RULE_SCHEMA`) shared by all three prompts, with "allowed" (only),
   "permitted" (fine for) and "forbidden" defined precisely.
2. A deterministic normaliser for real-world variants (SOV → verb-final, a tense table → one rule per ending,
   text → list, "subject is tavo" → agreement, an English term → the entry's Nadi-9 word). Every changed rule is marked
   `normalised`. **Grounding is never relaxed.**
3. Misfits become UNTESTABLE instead of being rejected, so the quoted claim stays visible.
4. Rate limits: the provider waits as long as Groq asks (Retry-After header, or "try again in 7.3s" in the message)
   and resends. A 429 was never processed, so waits do not spend the run budget.
5. Smaller packages: only the rules a scene needs, each as one readable line (3–4k tokens → 1.7–2.6k).
6. Cloudflare 1010: Groq's CDN blocks urllib's default User-Agent; the provider now sends its own.

**Run 1 replayed through the fixes (no new API calls):** 0 → 22 usable grammar rules, E09/E14 caught again.

**Run 2 (fresh, with the format spec):** 8 model calls, no failures, ~5 minutes (mostly rate-limit waits).
- Learning: 18/18 grammar rules grounded, 0 rejected; respect-2 returned directly as agreement
  (present: no evidence); VAR-1 correctly tagged `requires_tag: estranged`; E09 and E14 flagged.
- **The model proposed 7 rules of its own; the evidence refuted 4** ("na after the verb", "verb medial",
  "-sa marks future", and a misreading of VAR-2). This is the brief's "model confidently proposes an unsupported
  rule", caught for real.
- Translation: 24 proposals, and S017 declined for the right reason. The real output has real errors for Stage 5:
  verb not final (S019, S024, S005, S012), morpheme boundaries written with hyphens (`rov-a` where the examples write
  fused forms), the tavo/-sa ending missing (S022), respectful `tavora` from the headman to Arun (S023), "friend"
  silently dropped to avoid `sorai` (S012), a stray `?and` (S024, already caught structurally).

**Model availability:** the account has no `llama-3.3-70b-versatile`; `openai/gpt-oss-120b` is the default now.
Run `python -m nadi9 check-provider` to list the models a key can use.

# PART B3 — Audit against the brief, and the fixes

A requirement-by-requirement audit of the finished product found six gaps; all are fixed and tested (162 tests):
1. **Audio:** Groq Whisper transcriber (`tools/transcribe.py`), with the pack's vocabulary as a spelling hint.
   Two bugs found by a real test: transcripts were never ingested, and same-named files shared a source id.
   Machine transcripts without a human gloss are kept for reviewers and not learned from.
2. **PDF/Word:** install guidance in the README; a missing library produces a clear "pip install …" message (tested).
3. **The duplicated dispute question in S001:** each competing pair is recorded once; each rule is named once.
4. **Source scope:** captured from the manifest, the file header or a "Scope:" line, and shown in the report
   with author, date and status.
5. **Code-switching:** a measured rule (`induced:code_switching`), tested on the examples (supported, 4.46 for / 0
   against on Pack 1) and re-applied by the verifier (a loanword with an unknown ending is flagged).
6. **README:** says plainly what a new pack produces with and without a key.

# PART C — State and audit

- **RunState** holds the profile hash, lexicon claims, hypotheses, example audit, social graph, plan, decisions, dependency index, applied events, budget and a log. Every change increments `version` and writes a snapshot.
- **Each decision stores `state_version`,** so "what did the agent know when it decided S019?" = open snapshot N.
- **Everything is referenced.** A reviewer can go from any subtitle → its evidence refs → the exact row or sentence in the pack.
- **Trust boundary:** pack content is data, never instructions. Quarantined records never reach a prompt. Model output is always validated by code before it changes state.

# PART D — Build order and time

The brief suggests 10–12 hours in total, and S0–S1 (including the second pack) are done. The remaining work is prioritised so that stopping at any point still leaves a working pipeline.

| Order | Work | Estimate | Priority |
|---|---|---|---|
| 1 | Unify the run budget (gap 1) ✅ | 0.25 h | must |
| 2 | S2 hypotheses + example audit + S1 feedback pass ✅ | 2 h | must |
| 3 | S3 social graph, line features, risk plan ✅ | 1 h | must |
| 4 | S4 translator + replay fixtures for the episode ✅ | 1.5 h | must |
| 5 | S5 verifier V1–V7 ✅ | 1.5 h | must |
| 6 | S6 decider + questions ✅ | 1 h | must |
| 7 | S7 run loop, events, dependency index ✅ | 1.5 h | must |
| 8 | S8 outputs + `nadi9 run` + sample_run ✅ | 0.75 h | must |
| 9 | README, ARCHITECTURE, AI_COLLABORATION, KNOWN_LIMITATIONS ✅ | 1 h | must |
| 10 | Real provider (Groq) + `--record` ✅ · Claude optional | 1 h | should |
| 11 | V8 LLM critic; S2 model-proposed rules | 1 h | could |

If time runs short, cut from the bottom and list the cuts in `KNOWN_LIMITATIONS.md`.

# PART E — Known limitations (running list)

- Source-language handling assumes English (`english.py`, swappable).
- The synonym table is small and hand-written.
- The replay fixtures for the synthetic pack are hand-authored stand-ins for model output, labelled as such.
- In pure mock mode an unfamiliar file format ends up unclassified (safe, but a real model would classify it).
- Audio transcription is mocked; a real engine (e.g. local Whisper) fits behind the `Transcriber` interface.
- Confidence weights and thresholds are hand-set and tuned only on synthetic data.
- The status and relationship inference is keyword-based; every inference is exposed as an assumption for review.
