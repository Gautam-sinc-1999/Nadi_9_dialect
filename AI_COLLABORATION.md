# AI collaboration log

## Tools

| Tool | Role |
|---|---|
| **Claude Code** (Claude Opus 5.5, in VS Code) | coding assistant: read the brief, proposed the design, wrote the code, the synthetic packs, the tests and these documents, and ran everything |
| **Groq, `openai/gpt-oss-120b`** | not a coding assistant: the model *inside* the system (rule extraction, expert-claim extraction, translation). Its real output is recorded in `fixtures/recorded*/` |

## How the work was divided

The candidate set the direction and the constraints; the assistant proposed designs and did the implementation;
every stage was run and tested before moving on. Decisions that came from the candidate, not the assistant:

- **No official pack → build a synthetic one**, and say so openly.
- **"It should adapt to whatever pack is provided."** The assistant's first design implicitly assumed our own file
  formats. This requirement led to the pack profiler (S0) and to a second, deliberately different pack
  (`packs/variant_veshti/`) used to test every stage.
- **A written plan first** (`plan.md`): a high-level version for the interview, then each stage technically.
- **"Are you verifying after every stage?"** This question led to an explicit list of what the tests do and do *not*
  prove, and to the decision to run a real model as soon as possible, rather than trusting hand-written fixtures.
- **Use Groq instead of Claude** as the runtime model, with the key in `.env`; **do real runs**.

## How AI output was checked

1. **Every stage was run on two packs** (different language, formats and structure), not only the one it was built for.
2. **Tests against an answer key** written when the synthetic pack was made (poisoned entries, wrong examples,
   expected decisions, which lines each correction may and may not change), checked in both directions: nothing
   missed, and nothing extra.
3. **Real model runs**, recorded and kept as fixtures, so tests pin behaviour on output nobody wrote by hand.
4. **Reading the actual output, not only the pass/fail counts.** Several bugs below were found by reading a report
   whose tests were green.
5. **Determinism**: the suite is run under several `PYTHONHASHSEED` values, and a test checks that `sample_run/` is
   exactly what replaying the recorded run produces.

## What the AI got wrong, and how it was caught

### The biggest one: hand-written fixtures hid a format mismatch
The assistant wrote both the prompts and the replay fixtures that stand in for model output, and the fixtures matched
the code perfectly. **The first real run rejected all 23 grammar rules** the real model extracted. They were correct
and correctly quoted, but in the model's own format (`{"order": "SOV"}` instead of `{"verb_position": "final"}`),
because the prompts named the rule templates without spelling out their fields. The system failed safely (nothing was
guessed), but the two wrong examples were no longer caught. Fixes: one exact format spec shared by every prompt; a
deterministic normaliser for real-world variants, marked `normalised`; misfits kept as *untestable* instead of
discarded. Run 1's recordings are kept in `fixtures/recorded_run1_before_schema_fix/` as evidence.
**Lesson:** a verifier written by the same author as the thing it verifies shares its blind spots. This is also why
the S5 verifier uses no model.

### Found by the second pack
- A part-of-speech check matched "verb" inside "**ad**verb", so time words were counted as verbs.
- The column matcher let a fuzzy match (`comment` ≈ `commentid`) take a column before exact matches were tried.
- Field detection looked only at the first 20 rows, so a `notes` column present only on row 60 was missed, and that
  was exactly where the planted prompt injection was. Hidden columns are where injected text hides.
- A character window around a word missed "A yes/no **question** begins with the particle **ne**"; replaced by the
  whole sentence.

### Found by the real model
- Groq's CDN (Cloudflare) blocked Python's default User-Agent (error 1010), and the assistant's error message would
  have blamed the API key.
- The default model (`llama-3.3-70b-versatile`) was not available on the account; `check-provider` now lists models.
- Rate limits (8,000 tokens/minute) drained the budget through retries → circuit breaker; waits that don't spend
  budget; smaller evidence packages (3–4k → 1.7–2.6k tokens); halving the reply allowance on "413 request too large".
- The real model missed one grammar rule (polite commands), so a familiar command to a mother was approved →
  safety net: status-dependent forms with no tested rule go to review.
- The real model left "friend" untranslated to avoid an insult, and the review question then falsely said "no word
  exists". Fixed to say the word exists and why it was avoided.
- The real model wrote gaps as `?word`, which were read as invented words; now normalised.
- The real model silently skipped a line; now reported instead of left blank.

### Found by reading output whose tests passed
- False "status clash" warnings on every "you" line (choosing between pronouns is a choice, not a clash);
  "Thank **you**" treated as addressing someone; "meeting" stemmed to "meet" before the loanword check.
- `ama` is both "mother" and a character's name, so correct lines were flagged for dropping "mother".
- "has come back" read as present tense, producing false tense errors.
- The first replanning design re-translated a line just because a rule it used became *stronger*; now only changes to
  *what* a rule says trigger re-translation, and strength changes only re-check.
- **The humour check never fired.** Stage 3 labels humour "humour" and the decider's pattern did not include that
  word. It was found while checking a claim in `KNOWN_LIMITATIONS.md` against the code. Now fixed and pinned by a test.
- The order of one output list changed between processes (iteration over a `set`); found because a test compares the
  committed sample run with a fresh replay.

### Found by auditing the finished product against the brief
The candidate asked for a requirement-by-requirement audit (no edits), then for the gaps to be fixed:
- **Audio could not be transcribed**, although the brief's pack has 5 audio interviews. A Groq Whisper transcriber
  was added. Testing it with real (synthetic) speech found two more bugs: **transcribed segments were never
  ingested** (the code path stored them and stopped), and **an audio file sharing its name with its interview
  record got the same source id**, so it was transcribed twice and its segments duplicated.
- Whisper's output for Nadi-9 was confidently wrong (confidence 0.62, above the learning threshold). Machine
  transcripts without a human gloss are now kept for reviewers only.
- The sample's S001 question listed one expert's position twice, and a rule appeared to compete with itself.
- Source *scope* (named in the brief's first task) was read but never reported; it is now in the report.
- Code-switching (named in the brief's second task) was never a tested rule; it now is.
- PDF/Word support depended on libraries the README never mentioned.

### Tests that were wrong
- One test could never fail (it checked for a word that could not appear in that list); it was removed, not kept
  for the count.
- One test assumed the real model would phrase a gap exactly like the hand-written fixture; it was rewritten to
  assert what must hold in any run.
