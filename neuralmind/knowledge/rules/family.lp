% Kinship reasoning -- the smallest knowledge base that shows every feature of
% the engine: recursion, negation, comparisons and integrity constraints.
%
% Facts expected as input: parent/2, male/1, female/1, born/2.

person(X) :- parent(X, _).
person(Y) :- parent(_, Y).
person(X) :- male(X).
person(X) :- female(X).

father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).
child(Y, X) :- parent(X, Y).
son(Y, X) :- parent(X, Y), male(Y).
daughter(Y, X) :- parent(X, Y), female(Y).

% Recursion: an ancestor is a parent, or a parent of an ancestor.
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
descendant(Y, X) :- ancestor(X, Y).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
grandmother(X, Z) :- grandparent(X, Z), female(X).
grandfather(X, Z) :- grandparent(X, Z), male(X).

% A comparison keeps a person from being their own sibling.
sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y.
brother(X, Y) :- sibling(X, Y), male(X).
sister(X, Y) :- sibling(X, Y), female(X).
aunt(A, C) :- sibling(A, P), parent(P, C), female(A).
uncle(U, C) :- sibling(U, P), parent(P, C), male(U).
cousin(X, Y) :- parent(P1, X), parent(P2, Y), sibling(P1, P2).

% Negation: someone with no recorded parent is a root of the family tree.
has_parent(X) :- parent(_, X).
root_ancestor(X) :- person(X), not has_parent(X).

% Arithmetic over birth years.
older(X, Y) :- born(X, YearX), born(Y, YearY), YearX < YearY.

%@ Nobody can be their own ancestor
:- ancestor(X, X).

%@ Nobody has two different birth years
:- born(X, A), born(X, B), A != B.

%@ A person cannot be both male and female in this model
:- male(X), female(X).

%@ A parent must be born before their child
:- parent(X, Y), born(X, A), born(Y, B), A >= B.
