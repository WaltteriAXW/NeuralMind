% Phase 1 of the roadmap: the classic neurosymbolic toy task.
%
% A CNN classifies two digit images and asserts digit/2. Everything below is
% symbolic: the sum is computed, not learned, and is therefore exactly right
% whenever the perception is.
%
% Input facts: digit(Slot, Value) from the perception layer, and optionally
% expected_sum(S) as supervision the consistency layer can check against.

slot(d0). slot(d1).

sum(S) :- digit(d0, A), digit(d1, B), S = A + B.
carry(1) :- digit(d0, A), digit(d1, B), S = A + B, S > 9.
carry(0) :- digit(d0, A), digit(d1, B), S = A + B, S <= 9.
units(U) :- sum(S), U = S \ 10.

%@ Each image shows exactly one digit
:- digit(Slot, A), digit(Slot, B), A != B.

%@ A digit must be between 0 and 9
:- digit(_, V), V < 0.

%@ A digit must be between 0 and 9
:- digit(_, V), V > 9.

%@ The computed sum must match the expected sum
:- expected_sum(S), sum(T), S != T.
