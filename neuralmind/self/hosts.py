"""Eight unlabelled hosts, for asking the mind where it thinks it is.

P2.4's bar is that one ``Mind``, with no configuration, identifies the facets
of at least eight hosts it was never told about. These are those hosts.

Being honest about what each one is:

* **banking chat** and **out-of-scope** are real utterances, from BANKING77 and
  CLINC150's deliberate out-of-scope set. Fetch them with
  ``scripts/download_contexts.py``; without them those two streams fall back to
  a handful of written-out lines, and the tests say which they used.
* the rest are **synthesised in the shape of** the roadmap's hosts -- a BabyAI
  grid, a TextWorld transcript, a CadQuery parameter stream, a MultiWOZ-style
  service dialogue, a financial table feed, a grocery stock feed, and a pump
  system no facet pack covers. Each mirrors the *structure* of its original
  (keys, value formats, what moves between ticks) because structure is what
  the mind reads. None of them is the real corpus, and the docs say so rather
  than letting "eight hosts" sound like eight downloads.

The pump system is the important one: it is deliberately outside every facet
the profiles know, so the honest answer is a low-confidence reading and an
unresolved domain. A context discovery that confidently labels it has failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

__all__ = ["HOSTS", "stream", "host_names", "real_data_available"]

DATA = Path(__file__).resolve().parents[2] / "data" / "contexts"


def real_data_available() -> bool:
    return (DATA / "clinc150.json").exists() and (DATA / "banking77-test.csv").exists()


# -- synthesised hosts -----------------------------------------------------


def _grid_game(n: int = 8) -> list[dict]:
    """A BabyAI-shaped grid world: position, an action list, a reward, a tick."""
    out = []
    x, y = 1, 1
    for step in range(n):
        x += step % 2
        y += (step + 1) % 2
        out.append(
            {
                "step": step,
                "position": {"x": x, "y": y},
                "direction": ["north", "east", "south", "west"][step % 4],
                "actions": ["left", "right", "forward", "pickup", "toggle"],
                "carrying": "none" if step < 4 else "key",
                "reward": 0 if step < n - 1 else 1,
                "mission": "pick up the red key",
            }
        )
    return out


def _text_game(n: int = 6) -> list[str]:
    """A TextWorld-shaped transcript: second person, an inventory, exits."""
    lines = [
        "You are in a kitchen. There is a closed fridge here. Exits lead north and east.",
        "You open the fridge. Inside you see a carrot and a block of cheese.",
        "You take the carrot. Your inventory now holds a carrot and a rusty key.",
        "You go north. This is the pantry. A locked chest sits in the corner.",
        "You unlock the chest with the rusty key. It contains a recipe book.",
        "You read the recipe book. It calls for a carrot, diced.",
    ]
    return lines[:n]


def _cad_parameters(n: int = 6) -> list[dict]:
    """A CadQuery-shaped parameter stream: quantities with units, no money."""
    out = []
    for step in range(n):
        out.append(
            {
                "part": f"BRKT-{step:03d}",
                "wall_thickness": f"{2.0 + step * 0.25:.2f} mm",
                "hole_spacing": f"{12 + step} mm",
                "material": "aluminium" if step % 2 else "steel",
                "load": f"{3.2 + step * 0.1:.1f} kN",
                "tolerance": "0.05 mm",
            }
        )
    return out


def _service_dialogue(n: int = 6) -> list[str]:
    """A MultiWOZ-shaped service dialogue: a person asking for something."""
    lines = [
        "I need to book a table for four people on Thursday evening.",
        "Could you find somewhere in the centre that serves Italian food?",
        "What is the price range for that restaurant?",
        "Please book it for 19:30 and give me the reference number.",
        "Can you also find me a taxi from the hotel to the restaurant?",
        "Thank you, that is everything I needed today.",
    ]
    return lines[:n]


def _financial_tables(n: int = 5) -> list[list[dict]]:
    """A FinQA-shaped report feed: tables of periods and amounts."""
    out = []
    for step in range(n):
        out.append(
            [
                {
                    "period": f"2026-Q{quarter}",
                    "revenue": f"{1200 + quarter * 40 + step} USD",
                    "cost": f"{800 + quarter * 15 + step} USD",
                    "open": 101.4 + step,
                    "close": 103.9 + step,
                    "table": "income_statement",
                }
                for quarter in (1, 2, 3, 4)
            ]
        )
    return out


def _grocery_feed(n: int = 6) -> list[list[dict]]:
    """A grocery stock feed: SKUs, stock levels, prices, best-before dates."""
    products = [
        ("MLK-1L", "Whole milk 1 l", "1.29 EUR", "dairy"),
        ("BRD-500", "Rye bread 500 g", "2.49 EUR", "bakery"),
        ("EGG-12", "Free-range eggs 12", "3.95 EUR", "dairy"),
        ("APL-1K", "Apples 1 kg", "2.19 EUR", "produce"),
    ]
    out = []
    for step in range(n):
        out.append(
            [
                {
                    "sku": sku,
                    "product": name,
                    "category": category,
                    "price": price,
                    "stock": max(0, 20 - step * 3 - index * 2),
                    "reorder_at": 8,
                    "best_before": f"2026-09-{19 + index}",
                    "supplier": f"SUP-{index}",
                }
                for index, (sku, name, price, category) in enumerate(products)
            ]
        )
    return out


def _pump_system(n: int = 8) -> list[dict]:
    """A host no facet pack covers. The honest answer here is "unresolved"."""
    out = []
    for step in range(n):
        out.append(
            {
                "tick": step,
                "pump_a": "running" if step % 3 else "idle",
                "valve_b": "open" if step % 2 else "closed",
                "flow": f"{12 + step % 4} l/min",
                "pressure": f"{2.1 + 0.1 * (step % 5):.1f} bar",
                "alarm": step == 5,
            }
        )
    return out


def _money_transfer(n: int = 5) -> list[dict]:
    """A host that can move money, and offers the action that does it.

    Money plus an action list is what makes stakes high, and it has to be high
    before any domain is recognised -- the mind must not need to work out it
    is in a bank before it becomes careful.
    """
    out = []
    for step in range(n):
        out.append(
            {
                "account_id": f"ACC-{100 + step}",
                "customer_id": f"CUS-{200 + step}",
                "balance": f"{1500 - step * 120} EUR",
                "amount": f"{120 + step * 10} EUR",
                "counterparty": f"ACC-{300 + step}",
                "limit": 5000,
                "approved": step % 2 == 0,
                "actions": ["transfer", "hold", "reject"],
            }
        )
    return out


# -- real-data hosts --------------------------------------------------------


def _banking_chat(n: int = 8) -> list[str]:
    import csv

    path = DATA / "banking77-test.csv"
    if not path.exists():
        return [
            "Why was my card payment declined?",
            "Can I get a refund on that transfer?",
            "How long does a top-up take to show in my balance?",
        ][:n]
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["text"] for row in rows[:n]]


def _out_of_scope(n: int = 8) -> list[str]:
    path = DATA / "clinc150.json"
    if not path.exists():
        return [
            "how much has the dow changed today",
            "what is the airspeed of an unladen swallow",
        ][:n]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [text for text, _label in data["oos_test"][:n]]


#: name -> (builder, what the mind ought to notice)
HOSTS: dict[str, tuple] = {
    "grid_game": (_grid_game, {"agency", "spatial"}),
    "text_game": (_text_game, {"conversation"}),
    "cad_parameters": (_cad_parameters, {"quantities"}),
    "banking_chat": (_banking_chat, {"conversation", "money"}),
    "service_dialogue": (_service_dialogue, {"conversation"}),
    "financial_tables": (_financial_tables, {"money"}),
    "grocery_feed": (_grocery_feed, {"inventory", "catalog", "money"}),
    "pump_system": (_pump_system, {"quantities"}),
    "money_transfer": (_money_transfer, {"money", "parties"}),
}


def host_names() -> list[str]:
    return list(HOSTS)


def stream(name: str, n: Optional[int] = None) -> list:
    """The observations one host sends. Nothing names the host to the mind."""
    if name == "out_of_scope":
        return _out_of_scope(n or 8)
    if name not in HOSTS:
        raise KeyError(f"unknown host {name!r}; choose from {', '.join(HOSTS)}")
    builder = HOSTS[name][0]
    return builder(n) if n is not None else builder()
