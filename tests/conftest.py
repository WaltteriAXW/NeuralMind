import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a second")


@pytest.fixture
def family_kb():
    from neuralmind.knowledge.base import KnowledgeBase

    kb = KnowledgeBase("family").load_builtin("family")
    kb.add_facts(
        [
            "parent(maria, juho)", "parent(maria, liisa)", "parent(juho, aino)",
            "parent(aino, elias)", "female(maria)", "male(juho)", "female(liisa)",
            "female(aino)", "male(elias)", "born(maria, 1948)", "born(juho, 1972)",
            "born(liisa, 1975)", "born(aino, 1999)", "born(elias, 2024)",
        ]
    )
    return kb
