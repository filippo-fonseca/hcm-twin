"""Shared fixtures. Solves are cached per session because they are the expensive part."""

from __future__ import annotations

import pytest

from hcmtwin import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
    BeatResult,
    Observables,
    observe,
    simulate,
)


@pytest.fixture(scope="session")
def healthy_beat() -> BeatResult:
    return simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0, record_trace=True)


@pytest.fixture(scope="session")
def healthy(healthy_beat: BeatResult) -> Observables:
    return observe(healthy_beat, HEALTHY_GEOMETRY, RESTING_LOADING)


@pytest.fixture(scope="session")
def hcm_beat() -> BeatResult:
    return simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 0.0, record_trace=True)


@pytest.fixture(scope="session")
def hcm(hcm_beat: BeatResult) -> Observables:
    return observe(hcm_beat, HCM_GEOMETRY, RESTING_LOADING)
