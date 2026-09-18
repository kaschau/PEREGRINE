"""The bit gate: every case's interior state after its steps hashes to
what this platform's reference says. The references are a round's
starting point, recorded with PG_GATE_RECORD=1 and kept out of the repo:
a hash that moves during the round is evidence, not a verdict -- decide
on the merits, then re-record."""

import os
from pathlib import Path

import pytest
import yaml

from . import cases

references = Path(__file__).with_name("references.yaml")


def load():
    return yaml.safe_load(references.read_text()) if references.exists() else {}


@pytest.mark.parametrize("name", list(cases.cases))
def test_gate(my_setup, name):
    mb = cases.build(**cases.cases[name])
    cases.step(mb)
    got = cases.digest(cases.blockDigests(mb))
    platform = cases.platform()
    known = load()
    if os.environ.get("PG_GATE_RECORD"):
        known.setdefault(platform, {})[name] = got
        references.write_text(yaml.safe_dump(known, sort_keys=True))
        return
    if platform not in known or name not in known[platform]:
        pytest.skip(
            f"no reference for {name!r} on {platform}; record with PG_GATE_RECORD=1"
        )
    assert (
        got == known[platform][name]
    ), f"{name} on {platform}: {got} != {known[platform][name]}"
