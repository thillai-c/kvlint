"""Turn cache divergences into explained findings.

Pipeline (build plan section 5.3):

1. Attribute each request to the prefix it drifted from, token-exact.
2. Map the divergence token back to a character position via `char_offsets`.
3. Hand every rule the text on both sides of that point.
4. Aggregate each rule's hits into one finding for the whole log.
"""

from __future__ import annotations

from kvlint.lint.attribution import Attribution, attribute, shared_prefix_length
from kvlint.lint.context import CorpusContext, RuleContext
from kvlint.lint.rules import CORPUS_RULES, PER_DIVERGENCE_RULES
from kvlint.models import LintFinding, Request, Severity, TokenizedRequest

WINDOW_TOKENS = 64
"""How far either side of the divergence rules can see (build plan section 5.3)."""

_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def _char_pos(tokenized: TokenizedRequest, token_idx: int) -> int:
    """Character offset of a token index, clamped to the text."""
    offsets = tokenized.char_offsets
    if not offsets:
        return 0
    if token_idx >= len(offsets):
        return offsets[-1][1]
    return offsets[token_idx][0]


def _window(tokenized: TokenizedRequest, token_idx: int, tokens: int) -> tuple[str, str]:
    """Text before and after a token index, `tokens` wide on each side."""
    start = _char_pos(tokenized, max(0, token_idx - tokens))
    middle = _char_pos(tokenized, token_idx)
    end = _char_pos(tokenized, token_idx + tokens)
    text = tokenized.rendered_text
    return text[start:middle], text[middle:end]


def _system_span(request: Request, rendered: str) -> int:
    """Character offset where the system prompt stops.

    Located by searching for the system message content in the rendered text,
    which is exact because the template inserts it verbatim. Returns 0 when there
    is no system message, so nothing is treated as being inside one.
    """
    for message in request.messages:
        if message.role != "system" or not message.content:
            continue
        found = rendered.find(message.content)
        if found >= 0:
            return found + len(message.content)
    return 0


def _build_context(
    request: Request,
    tokenized: TokenizedRequest,
    prior: Request | None,
    prior_tokenized: TokenizedRequest | None,
    divergence: int,
    block_size: int,
) -> RuleContext:
    before, current_after = _window(tokenized, divergence, WINDOW_TOKENS)
    prior_before, prior_after = "", ""
    if prior_tokenized is not None:
        # Same token index in the prior request: everything before it is by
        # definition identical, so the two windows are directly comparable.
        prior_before, prior_after = _window(prior_tokenized, divergence, WINDOW_TOKENS)

    char_pos = _char_pos(tokenized, divergence)
    return RuleContext(
        request=request,
        tokenized=tokenized,
        prior=prior,
        prior_tokenized=prior_tokenized,
        divergence_token_idx=divergence,
        divergence_char_pos=char_pos,
        before=before,
        current_after=current_after,
        prior_after=prior_after,
        current_window=before + current_after,
        prior_window=prior_before + prior_after,
        block_size=block_size,
        in_system_prompt=char_pos < _system_span(request, tokenized.rendered_text),
    )


def run(
    requests: list[Request],
    tokenized: list[TokenizedRequest],
    block_size: int = 16,
) -> list[LintFinding]:
    """Lint a whole log, returning one finding per rule that fired."""
    if len(requests) != len(tokenized):  # pragma: no cover
        raise ValueError("requests and tokenized requests disagree in length")

    by_id = {r.request_id: (r, t) for r, t in zip(requests, tokenized, strict=True)}
    attributions = attribute(tokenized)

    # rule_id -> (request_ids that fired, first evidence seen, tokens lost)
    hits: dict[str, list[Attribution]] = {}
    evidence: dict[str, str] = {}

    for attribution, request, tokens in zip(attributions, requests, tokenized, strict=True):
        if attribution.divergence_token_idx is None:
            continue

        prior_pair = by_id.get(attribution.nearest_prior_request_id or "")
        prior, prior_tokenized = prior_pair if prior_pair else (None, None)

        context = _build_context(
            request,
            tokens,
            prior,
            prior_tokenized,
            attribution.divergence_token_idx,
            block_size,
        )

        for rule in PER_DIVERGENCE_RULES:
            found = rule.detect(context)
            if found is None:
                continue
            hits.setdefault(rule.RULE_ID, []).append(attribution)
            evidence.setdefault(rule.RULE_ID, found)

    findings = [
        LintFinding(
            rule_id=rule.RULE_ID,
            severity=rule.SEVERITY,
            affected_request_ids=[a.request_id for a in hits[rule.RULE_ID]],
            affected_fraction=len(hits[rule.RULE_ID]) / len(requests),
            # Upper bound: some tokens after a divergence were genuinely new and
            # could never have been cached. Reported as such (plan decision D7).
            est_tokens_lost=sum(a.tokens_lost for a in hits[rule.RULE_ID]),
            evidence=evidence[rule.RULE_ID],
            suggestion=rule.SUGGESTION,
            auto_fixable=rule.AUTO_FIXABLE,
        )
        for rule in PER_DIVERGENCE_RULES
        if rule.RULE_ID in hits
    ]

    corpus_context = CorpusContext(
        requests=requests,
        tokenized=tokenized,
        divergences={a.request_id: a.divergence_token_idx for a in attributions},
        shared_prefix_tokens=shared_prefix_length(tokenized),
        block_size=block_size,
    )
    for rule in CORPUS_RULES:
        hit = rule.detect(corpus_context)
        if hit is None:
            continue
        findings.append(
            LintFinding(
                rule_id=rule.RULE_ID,
                severity=rule.SEVERITY,
                affected_request_ids=hit.affected_request_ids,
                affected_fraction=(
                    len(hit.affected_request_ids) / len(requests) if requests else 0.0
                ),
                est_tokens_lost=hit.est_tokens_lost,
                evidence=hit.evidence,
                suggestion=rule.SUGGESTION,
                auto_fixable=rule.AUTO_FIXABLE,
            )
        )

    # Most expensive first, so the top of the table is where to start.
    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], -f.est_tokens_lost, f.rule_id))
    return findings


def worst_severity(findings: list[LintFinding]) -> Severity | None:
    """Highest severity present, for the CLI's `--fail-on` gate."""
    if not findings:
        return None
    return min((f.severity for f in findings), key=lambda s: _SEVERITY_ORDER[s])


def exceeds(findings: list[LintFinding], threshold: Severity) -> bool:
    """Whether any finding is at or above `threshold`."""
    limit = _SEVERITY_ORDER[threshold]
    return any(_SEVERITY_ORDER[f.severity] <= limit for f in findings)
