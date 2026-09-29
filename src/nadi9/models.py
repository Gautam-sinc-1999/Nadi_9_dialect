"""Canonical models. Every pack, whatever its format, is normalised into these.

Every record carries a `ref` of the form "<source_id>:<native_id>" so any claim can be
traced back to the exact place in the pack it came from.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    DICTIONARY = "dictionary"
    EXAMPLES = "examples"
    GRAMMAR = "grammar"
    EXPERT_NOTES = "expert_notes"
    FEEDBACK = "feedback"
    INTERVIEW = "interview"
    EPISODE = "episode"
    CHARACTERS = "characters"
    EVENT = "event"
    MANIFEST = "manifest"
    DOCUMENTATION = "documentation"
    UNCLASSIFIED = "unclassified"


class SourceStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"          # parsed, but some records failed validation
    UNAVAILABLE = "unavailable"  # could not be read or transcribed
    UNCLASSIFIED = "unclassified"


class Record(BaseModel):
    ref: str
    source_id: str
    native_id: str
    flags: list[str] = Field(default_factory=list)  # e.g. "quarantined:injection"
    raw: dict[str, Any] = Field(default_factory=dict)


class LexEntry(Record):
    term: str
    gloss: str
    pos: Optional[str] = None
    notes: Optional[str] = None


class Example(Record):
    source_text: str
    target_text: str
    speaker: Optional[str] = None
    addressee: Optional[str] = None
    relationship: Optional[str] = None
    context: Optional[str] = None
    status: Optional[str] = None


class TextSection(Record):
    """A citable chunk of free text: a grammar rule section, a paragraph, a note."""
    heading: Optional[str] = None
    text: str


class Note(Record):
    """An expert note or a viewer comment."""
    author: Optional[str] = None
    author_role: Optional[str] = None
    date: Optional[str] = None
    text: str
    cited_refs: list[str] = Field(default_factory=list)
    likes: Optional[int] = None


class Segment(Record):
    """One transcribed stretch of an interview."""
    text: Optional[str] = None
    gloss: Optional[str] = None
    asr_confidence: Optional[float] = None
    alternatives: list[str] = Field(default_factory=list)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class Character(Record):
    name: str
    description: Optional[str] = None
    age: Optional[int] = None


class EpisodeLine(Record):
    line_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: Optional[str] = None
    addressee: Optional[str] = None
    scene: Optional[str] = None
    notes: Optional[str] = None


class RawEvent(Record):
    """An event file, kept as-is. Interpreted later by the replanner."""
    text: str


# ---------- Rules ----------

class RuleTemplate(str, Enum):
    """The fixed set of rule shapes code knows how to test."""
    WORD_ORDER = "word_order"              # params: {"verb_position": "final"|"initial"}
    AFFIX = "affix"                        # params: {"kind": "suffix"|"prefix", "form": "en", "feature": "tense=past", "attaches_to": "verb"}
    PARTICLE_POSITION = "particle_position"  # params: {"particle": "na", "relative_to": "verb"|"clause_end"|"clause_start", "position": "before"|"after"|"at", "feature": "negation"}
    AGREEMENT = "agreement"                # params: {"if_word": "tavo", "then_affix": "sa", "on": "verb", "tenses": [...]}
    LEXICAL_CHOICE = "lexical_choice"      # params: {"term": "tavo", "condition": "addressee higher status"}
    CODE_SWITCHING = "code_switching"      # params: {"seen": ["office", "meeting"]}: English words used unchanged
    GAP = "gap"                            # the source explicitly says something is undocumented
    UNTESTABLE = "untestable"              # a real rule that fits no template


class RuleStatus(str, Enum):
    GROUNDED = "grounded"                  # quote verified in source
    REJECTED_UNGROUNDED = "rejected_ungrounded"
    REJECTED_INVALID = "rejected_invalid"


class RuleClaim(BaseModel):
    rule_id: str
    source_ref: str                        # e.g. "grammar:respect-2"
    template: RuleTemplate
    params: dict[str, Any] = Field(default_factory=dict)
    quote: str
    status: RuleStatus
    method: str                            # "llm" | "replay" | "heuristic"
    reason: Optional[str] = None


# ---------- Profile ----------

class FieldMapping(BaseModel):
    canonical_field: str
    source_field: Optional[str]
    method: str                            # "synonym" | "fuzzy" | "llm" | "manifest" | "missing"


class SourceProfile(BaseModel):
    source_id: str
    path: str
    format: str
    role: Role
    role_confidence: float
    role_method: str                       # "manifest" | "heuristic" | "llm" | "none"
    status: SourceStatus
    author: Optional[str] = None
    date: Optional[str] = None
    scope: Optional[str] = None           # what the source says it covers (manifest, or the file's own header)
    n_records: int = 0
    n_failed: int = 0
    field_mapping: list[FieldMapping] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AffixStat(BaseModel):
    form: str
    kind: str                              # "suffix" | "prefix"
    declared_by: list[str] = Field(default_factory=list)
    attested_in: list[str] = Field(default_factory=list)


class LanguageShape(BaseModel):
    """Facts measured from the pack. Nothing here is hard-coded."""
    affixes: list[AffixStat] = Field(default_factory=list)
    verb_position: dict[str, int] = Field(default_factory=dict)       # {"final": 17, "initial": 0, "medial": 2}
    verb_position_refs: dict[str, list[str]] = Field(default_factory=dict)
    particle_positions: dict[str, dict[str, int]] = Field(default_factory=dict)  # {"na": {"before_verb": 2}}
    unanalysed_tokens: dict[str, list[str]] = Field(default_factory=dict)       # token -> refs


class BudgetUsage(BaseModel):
    model_calls: int = 0
    tool_calls: int = 0
    max_model_calls: int = 0
    max_tool_calls: int = 0


class PackProfile(BaseModel):
    pack_id: str
    pack_dir: str
    provider: str
    sources: list[SourceProfile]
    rules: list[RuleClaim] = Field(default_factory=list)
    shape: LanguageShape = Field(default_factory=LanguageShape)
    quarantined: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    budget: BudgetUsage = Field(default_factory=BudgetUsage)


# ---------- Stage 1: lexicon claims and source trust ----------

class EntryOutcome(str, Enum):
    SUPPORTED = "supported"          # its meaning matches how the word is actually used
    CONTRADICTED = "contradicted"    # the word is used, a rival meaning matches, this one does not
    UNTESTED = "untested"            # the word never appears in usable evidence


class SenseCheck(BaseModel):
    """One dictionary entry's meaning, checked against the evidence."""
    entry_ref: str
    gloss: str
    head_words: list[str]
    qualifier_words: list[str]
    usage_hits: list[str] = Field(default_factory=list)      # examples/segments where the meaning fits
    usage_misses: list[str] = Field(default_factory=list)    # places the word is used but the meaning does not fit
    grammar_hits: list[str] = Field(default_factory=list)    # grammar sections that state this meaning
    qualifier_hits: list[str] = Field(default_factory=list)
    score: float = 0.0
    outcome: EntryOutcome = EntryOutcome.UNTESTED


class ClaimStatus(str, Enum):
    CORROBORATED = "corroborated"          # meaning confirmed by usage or grammar
    AGREED_UNTESTED = "agreed_untested"    # dictionaries agree, but no usage or grammar to confirm
    RESOLVED_CONFLICT = "resolved_conflict"  # sources disagreed, evidence picked a side
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    SINGLE_SOURCE = "single_source"        # one dictionary says so, nothing else confirms
    UNSUPPORTED = "unsupported"            # used in evidence but no listed meaning fits


class LexClaim(BaseModel):
    term: str
    kind: str                              # "word" | "affix"
    status: ClaimStatus
    meaning: Optional[str] = None          # the gloss we will use, if any
    chosen_entry: Optional[str] = None
    senses: list[SenseCheck]
    conflict: Optional[str] = None         # human-readable description
    discussed_in: list[str] = Field(default_factory=list)   # expert notes / feedback that mention the term
    reason: str


class SourceTrust(BaseModel):
    source_id: str
    role: Role
    prior: float
    supported: int = 0
    contradicted: int = 0
    untested: int = 0
    trust: float
    measured: bool                         # False = prior only, not yet tested against evidence
    conflicts_of_interest: list[str] = Field(default_factory=list)
    reason: str


# ---------- Stage 2: hypotheses and example audit ----------

class HypStatus(str, Enum):
    SUPPORTED = "supported"      # >= 2 weighted support, counterexamples <= 20%
    WEAK = "weak"                # some support, no counterexamples, not enough to rely on
    CONTESTED = "contested"      # real support and real counterexamples
    UNSUPPORTED = "unsupported"  # no evidence either way
    REFUTED = "refuted"          # counterexamples outweigh support
    UNTESTED = "untested"        # a gap, or a rule no checker can test


class Hypothesis(BaseModel):
    hyp_id: str
    origin: str                          # grammar | induced | expert | lexicon_note | model
    source_refs: list[str]
    template: RuleTemplate
    params: dict[str, Any] = Field(default_factory=dict)
    quote: Optional[str] = None
    author: Optional[str] = None
    conflict_of_interest: Optional[str] = None
    support: list[str] = Field(default_factory=list)
    counter: list[str] = Field(default_factory=list)
    support_w: float = 0.0
    counter_w: float = 0.0
    by_feature: dict[str, dict[str, float]] = Field(default_factory=dict)
    confidence: float = 0.5
    status: HypStatus = HypStatus.UNTESTED
    competes_with: list[str] = Field(default_factory=list)
    note: Optional[str] = None
    # --- changes over the run (Stage 7) ---
    version: int = 1
    history: list[dict[str, Any]] = Field(default_factory=list)       # earlier params, with the event that changed them
    external_support: dict[str, list[str]] = Field(default_factory=dict)  # feature -> refs (e.g. a linguist's correction)
    resolved_by: Optional[str] = None                                 # for gaps answered by an event


class ExampleAudit(BaseModel):
    example_ref: str
    suspicion: float
    suspect: bool
    weight: float                        # how much this example counts as evidence afterwards
    violations: list[dict[str, Any]] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    reason: str


# ---------- Stage 3: episode analysis and plan ----------

class SocialLink(BaseModel):
    """An *assumption* about who is talking to whom. Always shown to reviewers."""
    speaker: str
    addressee: str
    relation: str                        # sibling | parent_child | employer | friend | authority | group | unknown
    status: str                          # addressee relative to speaker: higher | equal | lower | elder_sibling | mixed | unknown
    method: str                          # kinship | role | age | vocative | none
    reason: str
    refs: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    english: str
    term: Optional[str] = None
    status: str                          # a ClaimStatus value, or loanword | name | gap
    entry: Optional[str] = None


class LineAnalysis(BaseModel):
    line_id: str
    ref: str
    scene: Optional[str]
    speaker: Optional[str]
    addressee: Optional[str]
    addressee_method: str                # given | vocative | turn_taking | none
    text: str
    features: dict[str, Any]
    social: Optional[SocialLink] = None
    tags: list[str] = Field(default_factory=list)          # estranged / reconciled
    coverage: list[Candidate] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    weak_words: list[str] = Field(default_factory=list)
    sensitive: list[str] = Field(default_factory=list)
    grammar_gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    risk: float = 0.0
    risk_factors: list[str] = Field(default_factory=list)


class ScenePlan(BaseModel):
    scene: str
    lines: list[str]
    max_risk: float
    total_risk: float
    rationale: str


class EpisodePlan(BaseModel):
    episode: str
    order: list[ScenePlan]
    lines: list[LineAnalysis]
    social: list[SocialLink]
    limits: dict[str, Any]
    budget: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


# ---------- Stage 4: translation proposals ----------

class Token(BaseModel):
    form: str                            # as written in the proposal, e.g. "roven" or "[?boy]"
    term: Optional[str] = None           # lexicon word the model says it used, e.g. "rov"
    affixes: list[str] = Field(default_factory=list)
    ref: Optional[str] = None            # evidence the model cites for this word


class Proposal(BaseModel):
    line_id: str
    scene: Optional[str]
    nadi_9_text: Optional[str]           # None = the model declined, failed, or was unavailable
    tokens: list[Token] = Field(default_factory=list)
    rules_used: list[str] = Field(default_factory=list)   # "grammar:respect-2#1@present"
    assumptions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    model_notes: Optional[str] = None
    flags: list[str] = Field(default_factory=list)         # structural problems found by code
    status: str = "proposed"             # proposed | no_translation
    reason: Optional[str] = None
    provider: str = ""
    package_terms: list[str] = Field(default_factory=list)  # what the model was allowed to use
    original_text: Optional[str] = None  # the model's text, if a verified spelling fix replaced it


# ---------- Stage 5: verification ----------

class Issue(BaseModel):
    check: str                           # V1..V7
    severity: str                        # info | warning | error | critical
    message: str
    refs: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)   # structured facts (rule id, feature, term) for Stage 6


class Verification(BaseModel):
    line_id: str
    status: str                          # checked | not_translated
    checks: dict[str, str] = Field(default_factory=dict)   # V1 -> pass | info | warning | error | critical
    issues: list[Issue] = Field(default_factory=list)
    backgloss: str = ""                  # word-by-word English built from the lexicon, not from the model
    fused_form: Optional[str] = None     # suggested spelling when morpheme boundaries were written out
    tokens: list[dict[str, Any]] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)  # lex:<term>, rule:<id>@<feature>, social:<a>-><b>

    @property
    def worst(self) -> str:
        order = ["pass", "info", "warning", "error", "critical"]
        return max(self.checks.values(), key=order.index, default="pass")


# ---------- Stage 6: decisions ----------

class Decision(BaseModel):
    """One subtitle decision, in the brief's format plus audit fields."""
    subtitle_id: str
    source_text: str
    nadi_9_text: Optional[str]
    confidence: float
    confidence_reason: str
    decision: str                        # APPROVE | HUMAN_REVIEW | INSUFFICIENT_EVIDENCE
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    review_question: Optional[str] = None
    # --- extensions ---
    priority: Optional[str] = None       # critical | high | normal (for review items)
    other_questions: list[str] = Field(default_factory=list)
    suggested_text: Optional[str] = None # e.g. the fused spelling from V6
    factors: dict[str, float] = Field(default_factory=dict)
    checks: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    backgloss: str = ""
    start_ms: int = 0
    end_ms: int = 0
    speaker: Optional[str] = None
    addressee: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)
    state_version: int = 0


class CanonicalPack(BaseModel):
    lexicon: list[LexEntry] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    sections: list[TextSection] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    lines: list[EpisodeLine] = Field(default_factory=list)
    events: list[RawEvent] = Field(default_factory=list)
