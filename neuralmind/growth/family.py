"""The family domain the growth loop is measured on.

A fixture rather than a benchmark, so the benchmark script and the school can
both use it without either importing the other. A shared measurement whose
inputs live inside one of its callers is a measurement that changes when that
caller does.

Big enough to matter. With five people, genuinely different definitions derive
identical atoms -- ``sibling`` and a piece of nonsense that happens to agree on
every pair there is -- and "settled" then means "the data cannot tell", which
is a fact about the data rather than about the loop. Nine people and nine
parent edges is enough to separate them.
"""

from __future__ import annotations

__all__ = ["FACTS", "SIBLING", "TARGETS"]

#: Nine people, three generations, both sexes on each.
FACTS = """
parent(maria, juho). parent(maria, liisa). parent(matti, juho). parent(matti, liisa).
parent(juho, aino).  parent(juho, eero).  parent(liisa, sanna).
parent(aino, taavi). parent(sanna, venla).
male(matti). male(juho). male(eero). male(taavi).
female(maria). female(liisa). female(aino). female(sanna). female(venla).
"""

#: Supporting definition for the targets that build on it.
SIBLING = "sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y."

#: ``target -> (background it needs, the definition to be recovered)``.
#:
#: The four cover the shapes that matter: a plain conjunction, one needing a
#: disequality, one resting on a learned relation, and one that is recursive.
TARGETS: dict[str, tuple[str, str]] = {
    "grandparent/2": ("", "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."),
    "sibling/2": ("", SIBLING),
    "aunt/2": (SIBLING, "aunt(X, Y) :- parent(P, Y), sibling(X, P), female(X)."),
    "ancestor/2": (
        "",
        "ancestor(X, Y) :- parent(X, Y). "
        "ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).",
    ),
}
