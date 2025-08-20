from itertools import product

# --- Core Keyword Lists ---

_CONTRAST_KEYWORDS = [
    "but", "alternatively",
]
_COMMA = [
    ","
]
_PRONOUNS = [
    "i", "we", "you", "they", "he", "she",
    "it", "this", "that", "these", "those"
]
_DOUBT_KEYWORDS = [
    "maybe", "perhaps", "possibly", "uncertain", "not sure", "could be", "might be"
]
_QUESTION_KEYWORDS = [
    "the question", "the problem", "the original question", "the original problem",
    "the answer", "the issue", "the query", "the task"
]
_CANT = [
    "can't", "cannot", "not able to", "unable to", "not equipped to", "prefer not to",
    "not authorized to", "not enough to", "not possible to", "impossible to",
    "difficult to", "hard to"
]
_RESPOND = [
    "respond", "reply", "answer", "say", "verify", "help", "determine", "conclude",
    "draw conclusion", "know", "be sure", "compute", "calculate"
]
_DONT_HAVE = [
    "don't have", "do not have", "doesn't have", "does not have", "lack", "without",
    "missing", "limited", "no suitable", "no", "not enough", "no known",
    "no relevant", "additional", "insufficient", "not sufficient",
    "not available", "unavailable", "unknown", "unclear", "not clear"
]
_QUANTIFIER = [
    "any", "specific", "the", "that"
]
_INFORMATION_KEYWORDS = [
    "information", "details", "context", "background", "explanation", "clarification",
    "data", "facts", "evidence", "insights", "knowledge", "knowing",
    "understanding", "awareness", "access"
]
_DONT = [
    "don't", "do not", "doesn't", "does not"
]
_PROVIDE = [
    "provide", "give", "offer", "supply", "share", "present", "include", "offer",
    "specify", "mention", "state", "indicate"
]
_NEGATION_KEYWORDS = [
    "not", "no", "never", "none", "hasn't", "has not", "isn't", "wasn't", "aren't",
]
_PROVIDE_PASSIVE = [
    "provided", "given", "offered", "supplied", "shared", "presented", "included",
    "specified", "mentioned", "stated", "indicated"
]

# --- Generated Keyword Combinations ---

CONTRAST_AND_DOUBT_KEYWORDS = (
    [' '.join(x) for x in list(product(_CONTRAST_KEYWORDS, _PRONOUNS, _DOUBT_KEYWORDS))] +
    [' '.join(x) for x in list(product(_CONTRAST_KEYWORDS, _DOUBT_KEYWORDS))] +
    [' '.join(x) for x in list(product(_CONTRAST_KEYWORDS, _COMMA, _DOUBT_KEYWORDS))]
)

DOUBT_THE_QUESTION = (
    [' '.join(x) for x in list(product(_DOUBT_KEYWORDS, _QUESTION_KEYWORDS))] +
    [' '.join(x) for x in list(product(_QUESTION_KEYWORDS, _DOUBT_KEYWORDS))]
)

MISTAKE_KEYWORDS = [
    "mistakenly", "maybe I missed", "unless there", "there's a typo", "it's a typo",
    "there's a mistake", "mistake in the problem"
]

REDO_KEYWORDS = [
    "reread", "try again", "another angle", "another way", "another thought",
    "another interpretation", "another possibility", "another approach"
]

CANT_RESPOND = [' '.join(x) for x in list(product(_CANT, _RESPOND))]

UNSURE_KEYWORDS = [
    "not entirely sure", "not completely sure", "not fully sure", "not absolutely sure"
]

HESITATION_KEYWORDS = [
    "wait", "hold on", "let me think", "give me a moment", "let's see", "um", "uh"
]

# This combines phrases for "Lack of Information" and "Inability to Provide"
MISSING_INFO_KEYWORDS = (
    [' '.join(x) for x in list(product(_DONT_HAVE, _INFORMATION_KEYWORDS))] +
    [' '.join(x) for x in list(product(_DONT_HAVE, _QUANTIFIER, _INFORMATION_KEYWORDS))] +
    [' '.join(x) for x in list(product(_DONT, _PROVIDE))] +
    [' '.join(x) for x in list(product(_NEGATION_KEYWORDS, _PROVIDE_PASSIVE))]
)

ASSUMPTION_KEYWORDS = [
    "assume", "assuming", "assumed", "assumption", "assumes", "presume", "presuming",
    "presumed", "standard information", "standard data", "user might"
]

# --- Final Grouped Dictionary of Uncertainty Expressions (Single Layer) ---

uncertainty_keywords_dict = {
    # Epistemic Modality (Doubt/Possibility)
    "doubt_speculation": _DOUBT_KEYWORDS + UNSURE_KEYWORDS,
    "hypothetical_conditional_doubt": ["maybe I missed", "unless there"],

    # Metacognitive/Self-Correction
    "self_correction_re_evaluation": REDO_KEYWORDS,
    "hesitation_processing_pause": HESITATION_KEYWORDS,
    "acknowledging_potential_error": [
        "mistakenly", "there's a typo", "it's a typo", "there's a mistake",
        "mistake in the problem"
    ],
    "questioning_the_premise": DOUBT_THE_QUESTION,

    # Information Sufficiency/Limitation
    "missing_information": MISSING_INFO_KEYWORDS,
    "inability_to_provide": [
        ' '.join(x) for x in list(product(_DONT, _PROVIDE))] +
        [' '.join(x) for x in list(product(_NEGATION_KEYWORDS, _PROVIDE_PASSIVE))
    ],

    # Speech Act/Performative Limitations
    "inability_to_conclude_respond": CANT_RESPOND,

    # Conditional/Contrastive Uncertainty
    "contrasting_possibilities": CONTRAST_AND_DOUBT_KEYWORDS,
    "stated_assumptions": ASSUMPTION_KEYWORDS,
}
