#!/usr/bin/env python3
"""A mind that is never told where it is, working it out.

No argument names the host. The mind sees observations, reads their *shape*,
and derives which building blocks are in play — money, inventory, conversation
— with the ordinary engine, so every hypothesis is a derivation you can take
apart rather than a score you have to take on faith.

The ordering is the part worth watching: **stakes rise before a domain is
recognised**. Money plus an action list is dangerous whether or not the mind
has worked out it is in a bank, and waiting to be sure before becoming careful
would be exactly backwards.
"""

from neuralmind import Mind
from neuralmind.kernel import L3
from neuralmind.self import HOSTS, ContextDiscovery, stream

print(f"{'host':18} {'stakes':7} {'domain':18} {'cover':6} what it could not place")
print("-" * 92)
for host in HOSTS:
    discovery = ContextDiscovery()
    discovery.observe_all(stream(host))
    reading = discovery.reading
    unplaced = ", ".join(reading.unexplained[:3]) or "—"
    print(
        f"{host:18} {reading.stakes:7} {reading.domain:18} "
        f"{reading.covered:<6} {unplaced}"
    )

# -- one of them, in full --------------------------------------------------

print("\n" + "=" * 92)
discovery = ContextDiscovery()
discovery.observe_all(stream("grocery_feed"))
print(discovery.reading.describe())
print("\nthe evidence, facet by facet:")
for facet in discovery.reading.facets:
    print(f"  {facet.name:12} {facet.confidence:.2f}   {', '.join(facet.because())}")

print("\nand the derivation behind the strongest one:")
print(discovery.reading.facets[0].proof)

# -- caution arrives before recognition ------------------------------------

print("\n" + "=" * 92)
mind = Mind(granted=L3)
first = stream("money_transfer", 1)[0]
mind.observe(first)
reading = mind.context.reading
print(f"after ONE observation: stakes {reading.stakes}, "
      f"domain {reading.domain} at {reading.domain_confidence:.2f}")
print(f"  {mind.kernel.autonomy.describe()}")
print(f"  {mind.decide('transfer(500)', needs=2)}")

# -- and it notices the ground moving --------------------------------------

print("\n" + "=" * 92)
discovery = ContextDiscovery(drift_after=2)
discovery.observe_all(stream("grocery_feed", 4))
print(f"settled on: {', '.join(discovery.reading.names)}")
for index, payload in enumerate(stream("grid_game", 6), start=1):
    discovery.observe(payload)
    if discovery.drifting:
        print(f"the host changed under it — noticed after {index} observation(s)")
        break
print(f"drifting: {discovery.drifting}; the honest response is to re-read, not to act")

print("\n" + "=" * 92)
print(mind.self_report())
