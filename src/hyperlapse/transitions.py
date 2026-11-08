from __future__ import annotations

import dataclasses
import random

from .config import TransitionsConfig
from .models import Segment

MAX_EFFECT_FRACTION = 0.3


def effect_duration(effect: str, config: TransitionsConfig, segment_len_s: float) -> float:
    return min(config.params[effect]["duration_s"], MAX_EFFECT_FRACTION * segment_len_s)


def assign_transitions(segments: list[Segment], config: TransitionsConfig, rng: random.Random) -> list[Segment]:
    """Attach a random effect to the start of some segments (never the first: it has no cut)."""
    active = config.active_effects
    if not active or config.probability <= 0:
        return list(segments)

    out = list(segments)
    last_index: int | None = None
    last_effect: str | None = None
    for i in range(1, len(out)):
        if last_index is not None and i - last_index - 1 < config.min_gap_cuts:
            continue
        if rng.random() >= config.probability:
            continue
        pool = [e for e in active if e != last_effect] or active
        effect = rng.choice(pool)
        out[i] = dataclasses.replace(out[i], transition=effect)
        last_index, last_effect = i, effect
    return out
