#!/usr/bin/env python3
"""Driving the interactive session from Python.

``neuralmind shell`` is the same object read from a terminal. Because
``Shell.handle`` is a pure function from a line to a response, a session is
also a perfectly ordinary thing to script -- for a walkthrough like this one,
or for a test.
"""

from neuralmind.shell import Shell

SESSION = [
    ":load triples",                                    # contradiction rules
    "Bob is a cat.",
    "Alice is a dog.",
    "All cats are mammals.",
    "All dogs are mammals.",
    "If something is a mammal then it is warm blooded.",
    "If something is warm blooded and it is a cat then it purrs.",
    "Is Bob warm blooded?",
    "Does Bob purr?",
    "Does Alice purr?",                                 # no, and it says why not
    "Bob is not warm blooded.",                         # caught immediately
    ":retract not_attr(bob, warm_blooded)",
    ":model attr",
]

shell = Shell()
for line in SESSION:
    print(f"> {line}")
    response = shell.handle(line)
    if response:
        print(response)

print("\n" + "=" * 62)
print("The session is a knowledge base like any other, so everything else")
print("in the project applies to it. Cross-checking it against clingo:\n")
print(" ", shell.engine.cross_check().report())

print("\nAnd it can be written out as ASP to keep:\n")
print("\n".join("  " + line for line in shell.knowledge.to_asp().splitlines()[:8]))
print("  ...")
