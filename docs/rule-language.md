# The rule language

NeuralMind's knowledge bases are written in a subset of ASP/Datalog that
`clingo` accepts verbatim. That is deliberate: the same `.lp` file runs on the
proof-producing Python engine and on an industrial ASP solver, and
`cross_check()` verifies they agree.

## The subset the Python engine runs

```prolog
% Facts: ground atoms.
parent(alice, bob).
born(alice, 1950).
label("a quoted string").

% Rules: head :- body.
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).   % recursion is fine

% Default negation, which must be stratified.
has_parent(X)  :- parent(_, X).
root(X)        :- person(X), not has_parent(X).

% Comparisons: = != < <= > >=
sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y.

% Integer arithmetic: + - * / \ **  ("=" binds the result)
sum(A, B, S)  :- digit(A), digit(B), S = A + B.

% Integrity constraints: reject any model where the body holds.
%@ Nobody can be their own ancestor
:- ancestor(X, X).

#show ancestor/2.
#const limit = 10.
```

`%@ text` on the line before a rule names it. The name appears in violation
reports and proof trees, which is the difference between

```
access_policy.lp:97 violated by: toxic_combination(erik, auditor, engineer)
```

and

```
Separation of duties: one person holds two conflicting roles
  violated by: toxic_combination(erik, auditor, engineer)
```

## What goes to clingo instead

Choice rules, aggregates, disjunctive heads, optimisation, intervals and weak
constraints are parsed but kept verbatim, and `Program.requires_asp` says which
feature was found. The Python engine refuses such a program with a pointer to
the backend rather than mis-running it:

```python
engine = ReasoningEngine(program, backend="clingo")
answer_sets = engine.answer_sets(limit=0)   # enumerate all of them
```

Use clingo for anything combinatorial — scheduling, assignment, puzzles,
optimisation. Use the Python engine when you need the proof.

## The two static checks

Both run when a program is loaded, and both catch real bugs before they become
wrong answers.

**Variable safety.** Every variable must be bindable by a positive body literal
or an assignment. These are all rejected:

```prolog
p(X, Y) :- q(X).              % Y is unbound in the head
p(X)    :- q(X), not r(Y).    % Y cannot be enumerated under negation
p(X)    :- q(X), X < Y.       % Y never gets a value
```

This caught a bug in this repository's own access-policy rules
(`blocked(E, _Res) :- suspended(E).` — the resource was never bound), which
would otherwise have silently derived nothing.

**Stratified negation.** Negation must not be recursive, or the program has no
unique least model:

```prolog
a(X) :- d(X), not b(X).
b(X) :- d(X), not a(X).       % rejected, and the cycle is named
```

Recursion through *positive* literals is fine and is what makes `ancestor`
work.

## Writing rules well

- **Name your constraints.** A `%@` label turns a line number into a sentence
  a domain expert can check.
- **Put the selective literal first.** The body is matched left to right and
  indexed on bound arguments; `parent(P, X), parent(P, Y)` is much cheaper than
  the other order when `P` is bound.
- **Expose denial reasons as predicates.** `denial_reason(Q, insufficient_clearance)`
  costs one rule and makes "why not?" answerable without reading the engine's
  diagnosis.
- **Keep arithmetic bounded.** `p(Y) :- p(X), Y = X + 1.` derives infinitely
  many atoms. The engine stops with a `ReasoningLimit` naming the rule rather
  than hanging, but the fix is a bound in the rule.
- **Test the rules before adding perception.** The blueprint is right that
  debugging a wrong rule is much easier before a noisy perception layer is in
  the loop. `neuralmind check -r yourrules.lp -F yourfacts.lp` is the loop.
