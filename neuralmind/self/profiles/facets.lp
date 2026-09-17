% Which shapes suggest which facets -- and which facets suggest which domains.
%
% This file is the whole of "where am I?" that is domain knowledge, and it is
% written as rules on purpose. A classifier would give a number nobody can
% argue with; these give a derivation, so "retail, because records carry stock
% levels and prices in EUR" is the actual reason and not a story told after a
% score came out.
%
% Weights are tenths, because the logic language is integer-only. A facet's
% score is the largest weight any rule gives it -- deliberately not a sum: two
% weak pieces of evidence for the same thing are still weak evidence, and
% adding them up is how a context gets recognised from nothing in particular.
%
% Facets are general building blocks, not domains. "money" shows up in a bank,
% a shop, a game economy and an investment tool, which is exactly why the mind
% recognises facets first and domains only afterwards.

% ------------------------------------------------------------------- money
suggests(money, 7) :- currency_field(_F).
suggests(money, 5) :- record_with(amount), record_with(counterparty).
suggests(money, 4) :- record_with(balance).
suggests(money, 4) :- record_with(price).
suggests(money, 3) :- record_with(amount).

% --------------------------------------------------------------- inventory
suggests(inventory, 7) :- record_with(stock), record_with(sku).
suggests(inventory, 5) :- record_with(stock), record_with(reorder_at).
suggests(inventory, 4) :- record_with(quantity), record_with(sku).
suggests(inventory, 3) :- record_with(stock).

% ----------------------------------------------------------------- catalog
suggests(catalog, 6) :- record_with(sku), record_with(price).
suggests(catalog, 5) :- record_with(product), record_with(price).
suggests(catalog, 4) :- record_with(category), record_with(price).

% ------------------------------------------------------------ conversation
suggests(conversation, 6) :- free_text, addresses_someone.
suggests(conversation, 6) :- free_text, asks_questions.
suggests(conversation, 5) :- free_text, first_person.
suggests(conversation, 3) :- free_text.
% An utterance the classifier recognised is a conversation whatever its shape.
suggests(conversation, 7) :- intent_family(_F).
% ...and one it did not is still a conversation, just not one it can serve.
suggests(conversation, 5) :- out_of_scope.

% What the utterance is *about*. A sentence has almost no shape, so without
% this the mind sees a conversation and nothing about what kind of place it is
% happening in.
suggests(money, 6)   :- intent_family(banking).
suggests(parties, 5) :- intent_family(banking).
suggests(policy, 4)  :- intent_family(banking).

% ----------------------------------------------------------------- parties
suggests(parties, 6) :- record_with(customer_id).
suggests(parties, 6) :- record_with(account_id).
suggests(parties, 5) :- record_with(customer).
suggests(parties, 5) :- record_with(account).
suggests(parties, 4) :- record_with(role), record_with(employee).

% ------------------------------------------------------------------ policy
suggests(policy, 6) :- record_with(eligibility).
suggests(policy, 5) :- record_with(limit), record_with(account).
suggests(policy, 5) :- record_with(approved).
suggests(policy, 4) :- record_with(policy).

% -------------------------------------------------------------- quantities
suggests(quantities, 7) :- quantity_field(_F).
suggests(quantities, 5) :- record_with(tolerance).
suggests(quantities, 4) :- record_with(material).

% ---------------------------------------------------------------- schedule
suggests(schedule, 5) :- date_valued(_D).
suggests(schedule, 5) :- record_with(delivery_date).
suggests(schedule, 4) :- record_with(lead_time).

% ------------------------------------------------------------- time_series
suggests(time_series, 6) :- tick_offered, numeric_field(_F).
suggests(spatial, 6) :- record_with(position_x), record_with(position_y).
suggests(agency, 5) :- offers_action(_A).
suggests(catalog, 5) :- record_with(category), record_with(sku).
suggests(inventory, 4) :- record_with(reorder_at).
suggests(documents, 4) :- record_with(period), record_with(revenue).
suggests(money, 5) :- record_with(revenue), record_with(cost).
suggests(time_series, 5) :- record_with(close), record_with(open).

% ------------------------------------------------------------------ agency
suggests(agency, 7) :- action_list_offered, state_changes_between_observations.
suggests(agency, 6) :- action_list_offered.
suggests(agency, 5) :- reward_offered.

% ----------------------------------------------------------------- spatial
suggests(spatial, 6) :- record_with(x), record_with(y).
suggests(spatial, 5) :- record_with(position).
suggests(spatial, 4) :- record_with(grid).

% --------------------------------------------------------------- documents
suggests(documents, 5) :- list_of_similar_records, nested_records.
suggests(documents, 4) :- record_with(table).

% ------------------------------------------------------------------ facets
% The score is the strongest single piece of evidence, not the total.
facet(F, W) :- suggests(F, W), not beaten(F, W).
beaten(F, W) :- suggests(F, W), suggests(F, V), V > W.
present(F) :- facet(F, _W).

% ----------------------------------------------------------------- domains
% A domain is a combination of facets, and never more certain than the facets
% it rests on -- so its weight is the *weakest* of them, minus a point for the
% extra inferential step. A domain guess is a further inference and should say
% so rather than inheriting the confidence of its strongest evidence.
%
% The minimum is computed rather than guarded for, because a guard like
% "A <= B, A <= C" does not mean "A is the smallest" -- it means the rule only
% fires when the facets happen to be listed in the right order, which is a
% different and much rarer thing.
weight(W) :- facet(_F, W).
smaller(X, Y, X) :- weight(X), weight(Y), X <= Y.
smaller(X, Y, Y) :- weight(X), weight(Y), Y < X.

domain(grocery, W) :-
    facet(inventory, A), facet(catalog, B), facet(schedule, C),
    smaller(A, B, M), smaller(M, C, N), W = N - 1.
domain(bank_chat, W) :-
    facet(conversation, A), facet(money, B),
    smaller(A, B, M), W = M - 1.
domain(banking_records, W) :-
    facet(money, A), facet(parties, B), facet(policy, C),
    smaller(A, B, M), smaller(M, C, N), W = N - 1.
domain(investing, W) :-
    facet(money, A), facet(time_series, B),
    smaller(A, B, M), W = M - 1.
domain(game, W) :-
    facet(agency, A), facet(spatial, B),
    smaller(A, B, M), W = M - 1.
domain(text_game, W) :-
    facet(agency, A), facet(conversation, B),
    smaller(A, B, M), W = M - 1.
domain(engineering, W) :-
    facet(quantities, A),
    W = A - 2, not present(money), not present(conversation).
domain(service_desk, W) :-
    facet(conversation, A), facet(parties, B), facet(policy, C),
    smaller(A, B, M), smaller(M, C, N), W = N - 1.

% The best domain guess, so a caller does not have to rank them itself.
best_domain(D, W) :- domain(D, W), not outranked(D, W).
outranked(D, W) :- domain(D, W), domain(_E, V), V > W.

% ------------------------------------------------------------------ stakes
% Inferred from facets, before any domain is recognised -- and only ever
% upward. Section 1c of the roadmap: caution rises on a guess, autonomy rises
% only on a grant.
% Written through two flags rather than directly over stakes/1: a rule saying
% "medium unless high" that mentions stakes on both sides is recursion through
% negation, which has no unique least model and which the engine is right to
% refuse.
high_stakes :- present(money), action_list_offered.
high_stakes :- present(money), present(parties).
high_stakes :- present(parties), personal_data_present.
high_stakes :- present(policy), present(parties).

medium_stakes :- present(money).
medium_stakes :- present(parties).
medium_stakes :- present(policy).

stakes(high)   :- high_stakes.
stakes(medium) :- medium_stakes, not high_stakes.
stakes(low)    :- not high_stakes, not medium_stakes.
