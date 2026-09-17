"""Working out where it is, and what it can do there.

The mind is never told which host it is running in. That has been a design
rule since P2.0 -- ``Mind.__init__`` takes no domain argument and a test
asserts it never will -- and this is the package that makes the rule
survivable rather than merely principled.

Four parts:

:mod:`~neuralmind.self.signature`
    Any payload into neutral facts about its *shape*. "Records carry a field
    named stock", never "this is a shop". Whether those add up to a shop is a
    question for rules, which can be argued with.
:mod:`~neuralmind.self.intent`
    A TF-IDF nearest-centroid classifier over BANKING77 and CLINC150, because
    a single sentence has almost no shape. It classifies and never generates,
    and its most useful output is "none of these", which is what CLINC150's
    out-of-scope set exists to teach.
:mod:`~neuralmind.self.context`
    The facet rules, run with the ordinary engine, so every hypothesis is a
    derivation. Facets first and domains second; stakes before either.
:mod:`~neuralmind.self.model`
    Facts about the mind itself, in the same vocabulary as everything else --
    which is the whole of what "self-awareness" means here, and why it is
    testable.

The ordering matters more than any of the parts. **Stakes are raised before a
domain is recognised**: money plus an action list is dangerous whether or not
the mind has worked out it is in a bank, and waiting for recognition before
becoming careful would be exactly backwards.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .context import PROFILES, UNRESOLVED, ContextDiscovery, Facet, Reading
from .hosts import HOSTS, host_names, real_data_available, stream
from .intent import OUT_OF_SCOPE, IntentClassifier, Prediction, numpy_available
from .model import Calibration, Competence, SelfModel
from .signature import Signature, SignatureReader, signature_of

__all__ = [
    "ContextDiscovery",
    "Reading",
    "Facet",
    "UNRESOLVED",
    "PROFILES",
    "SelfModel",
    "Competence",
    "Calibration",
    "Signature",
    "SignatureReader",
    "signature_of",
    "IntentClassifier",
    "Prediction",
    "OUT_OF_SCOPE",
    "numpy_available",
    "HOSTS",
    "host_names",
    "stream",
    "real_data_available",
]
