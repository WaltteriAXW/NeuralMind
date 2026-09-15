% Phase 6 domain pilot: an access-control policy checker.
%
% Why this domain. The blueprint asks for something "narrow but real", and
% access control is the case where this architecture beats a statistical model
% outright: the rules are written down already, every decision has to be
% auditable, a wrong answer is a security incident, and there is no training
% data to speak of. The proof tree is not a nice-to-have here -- it is the
% access-review evidence an auditor asks for.
%
% Input facts
%   employee(E).                    person(al)
%   has_role(E, R).                 direct role assignment
%   senior_to(R1, R2).              role hierarchy (R1 inherits R2's grants)
%   grants(R, Action, Class).       what a role may do at a classification
%   resource(Res).  classification(Res, Class).
%   clearance(E, Class).            highest classification E may see
%   requires_training(Class, T).  completed_training(E, T).
%   conflicting_duties(R1, R2).     separation-of-duties pairs
%   request(Q, E, Action, Res).     an access request to decide
%   submitted(P, E).  approved(P, E).   payment/change workflow
%   suspended(E).                   account state
%
% Derived
%   may_access(E, Action, Res), granted(Q), denied(Q), denial_reason(Q, Why)

% ---------------------------------------------------------------- lattice
% Classification levels, least sensitive first.
level_rank(public, 0).
level_rank(internal, 1).
level_rank(confidential, 2).
level_rank(restricted, 3).

class_level(C) :- level_rank(C, _).

% Clearance dominates a classification when its rank is at least as high.
dominates(Held, Needed) :-
    level_rank(Held, HeldRank),
    level_rank(Needed, NeededRank),
    NeededRank <= HeldRank.

cleared(E, Res) :-
    clearance(E, Held),
    classification(Res, Needed),
    dominates(Held, Needed).

% ---------------------------------------------------------------- roles
% Role inheritance is transitive: a senior role carries every junior grant.
inherits(R1, R2) :- senior_to(R1, R2).
inherits(R1, R3) :- senior_to(R1, R2), inherits(R2, R3).

effective_role(E, R) :- has_role(E, R).
effective_role(E, R2) :- has_role(E, R1), inherits(R1, R2).

% A role's grant applies at its own classification and every lower one.
role_permits(R, Action, Class) :- grants(R, Action, Class).
role_permits(R, Action, Lower) :-
    grants(R, Action, Class),
    dominates(Class, Lower),
    class_level(Lower).

permitted_by_role(E, Action, Res) :-
    effective_role(E, R),
    classification(Res, Class),
    role_permits(R, Action, Class).

% ---------------------------------------------------------------- blocks
% Training gates: a classification can require a course before anyone sees it.
missing_training(E, Res) :-
    classification(Res, Class),
    requires_training(Class, T),
    employee(E),
    not completed_training(E, T).

blocked(E, Res) :- missing_training(E, Res).
blocked(E, Res) :- suspended(E), resource(Res).

% ---------------------------------------------------------------- decision
may_access(E, Action, Res) :-
    permitted_by_role(E, Action, Res),
    cleared(E, Res),
    not blocked(E, Res).

granted(Q) :- request(Q, E, Action, Res), may_access(E, Action, Res).
denied(Q) :- request(Q, _E, _A, _R), not granted(Q).

% Denial reasons, so a refusal is as explainable as an approval.
denial_reason(Q, no_matching_role) :-
    request(Q, E, Action, Res),
    not permitted_by_role(E, Action, Res).
denial_reason(Q, insufficient_clearance) :-
    request(Q, E, _Action, Res),
    not cleared(E, Res).
denial_reason(Q, training_incomplete) :-
    request(Q, E, _Action, Res),
    missing_training(E, Res).
denial_reason(Q, account_suspended) :-
    request(Q, E, _Action, _Res),
    suspended(E).

% ---------------------------------------------------------------- audit
% Separation of duties, stated once and applied in both directions.
sod_conflict(R1, R2) :- conflicting_duties(R1, R2).
sod_conflict(R2, R1) :- conflicting_duties(R1, R2).

toxic_combination(E, R1, R2) :-
    effective_role(E, R1),
    effective_role(E, R2),
    sod_conflict(R1, R2).

self_approved(P, E) :- submitted(P, E), approved(P, E).

%@ Separation of duties: one person holds two conflicting roles
:- toxic_combination(_E, _R1, _R2).

%@ Four-eyes principle: nobody may approve their own submission
:- self_approved(_P, _E).

%@ A suspended account must not be granted access
:- granted(Q), request(Q, E, _A, _R), suspended(E).

%@ Access must never be granted above the requester's clearance
:- granted(Q), request(Q, E, _A, Res), not cleared(E, Res).

%@ Every resource must carry exactly one classification
:- classification(Res, A), classification(Res, B), A != B.

%@ A classification must be one of the four defined levels
:- classification(_Res, C), not class_level(C).

%@ The role hierarchy must not contain a cycle
:- inherits(R, R).
