"""Generate a demo log with known cache-breaking patterns.

The point of the demo is to show what kvlint finds on a workload whose problems
we already know, so the output can be checked rather than taken on faith. Every
anti-pattern here is one we have seen in real templates:

- a timestamp at the top of the system prompt, which is the single most common
  way to destroy a prefix cache
- a per-request correlation id, usually added for tracing
- the end user's email, added for personalization

All three sit **above** the stable instructions, which is what makes them
expensive: every token after them has to be recomputed.

Generation is seeded, so the same seed always produces the same log and the
demo's headline number is reproducible.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from kvlint.models import Message, Request

SYSTEM_PROMPT = """You are Acme Corp's customer support assistant.
Always answer in two sentences or fewer.
Cite the knowledge base article id you relied on, in square brackets.
Never speculate about delivery dates you cannot verify from the order record.
If the customer is angry, acknowledge the frustration before answering.
Escalate to a human when the customer asks for a refund above 500 dollars.
Do not disclose internal system names or ticket routing rules."""

QUESTIONS = [
    "Where is my order number {order}?",
    "I want to return order {order}, how do I start?",
    "Order {order} arrived damaged. What are my options?",
    "Can I change the delivery address on order {order}?",
    "Why was I charged twice for order {order}?",
    "Is order {order} eligible for express shipping?",
    "How do I cancel order {order}?",
    "When will order {order} be refunded?",
]

# Fixed so the demo is reproducible. Not the current clock: a demo whose output
# changes every run cannot have its headline number checked.
EPOCH = datetime(2026, 9, 11, 8, 0, 0, tzinfo=UTC)


def generate(count: int = 60, seed: int = 0, dirty: bool = True) -> list[Request]:
    """A synthetic support workload.

    With `dirty=False` the same workload is emitted without the injected
    anti-patterns, which is what the false-positive guard in the test suite uses.
    """
    rng = random.Random(seed)
    requests: list[Request] = []

    for i in range(count):
        system = SYSTEM_PROMPT
        if dirty:
            stamp = (EPOCH + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            system = (
                f"Current time: {stamp}\n"
                f"Request id: req_{rng.randrange(10**8):08d}f3\n"
                f"user: customer{i}@example.com\n" + system
            )

        order = 100_000 + i
        question = QUESTIONS[i % len(QUESTIONS)].format(order=order)

        requests.append(
            Request(
                request_id=f"demo-{i:04d}",
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=question),
                ],
            )
        )

    return requests
