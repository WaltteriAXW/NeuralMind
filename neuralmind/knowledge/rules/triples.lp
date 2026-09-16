% Consistency rules for the triple schema the text perception layer emits.
%
% The perception layer can assert both "X is blue" and "X is not blue" when it
% reads contradictory sources. Nothing in Datalog stops it -- these constraints
% do, and they report which pair of facts collided.

%@ Contradictory attribute
:- attr(X, A), not_attr(X, A).

%@ Contradictory class membership
:- isa(X, C), not_isa(X, C).

%@ Contradictory relation
:- rel(X, P, Y), not_rel(X, P, Y).

%@ Contradictory action
:- act(X, V), not_act(X, V).
