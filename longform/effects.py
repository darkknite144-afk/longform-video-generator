"""
Ken Burns motion effects: assignment and zoompan filter expression builder.

assign_effects() round-robins through EFFECTS so the motion varies scene to
scene instead of repeating the same slow zoom-in for the whole video.

_zoompan_expr() returns (z, x, y) expressions for ffmpeg's zoompan filter,
in terms of 'on' (output frame number, 0-indexed).
"""

from __future__ import annotations

import random

from .config import EFFECTS, PAN_ZOOM, ZOOM_MAX, ZOOM_PAN_COMBO_MAX


def assign_effects(n_scenes: int) -> list[str]:
    """Round-robin through EFFECTS for *n_scenes* scenes.

    Reshuffles each time the pool empties and swaps away an accidental
    back-to-back repeat at the cycle boundary. Seeded for reproducible
    output across re-runs.
    """
    rng = random.Random(1337)
    assigned: list[str] = []
    pool: list[str] = []
    for _ in range(n_scenes):
        if not pool:
            pool = EFFECTS.copy()
            rng.shuffle(pool)
            if assigned and pool[0] == assigned[-1]:
                swap_at = rng.randrange(1, len(pool))
                pool[0], pool[swap_at] = pool[swap_at], pool[0]
        assigned.append(pool.pop(0))
    return assigned


def _lerp(a: str, b: str, tf: int) -> str:
    """Linear interpolation expression: a + (b - a) * on / tf."""
    return f"({a}+(({b})-({a}))*on/{tf})"


def _centered(z: str) -> tuple[str, str]:
    """Return centered (x, y) expressions for a given zoom *z*."""
    return f"(iw/2-(iw/({z})/2))", f"(ih/2-(ih/({z})/2))"


def zoompan_expr(effect: str, total_frames: int) -> tuple[str, str, str]:
    """Return (z, x, y) expressions for ffmpeg's zoompan filter.

    x/y are written using the SAME closed-form zoom expression as z=,
    substituted in literally — NOT zoompan's built-in 'zoom' runtime
    variable. zoompan evaluates x/y against the *previous* output
    frame's 'zoom', one frame behind the z= it just computed for the
    current frame. With a zoom level that changes every single frame
    (every effect below except the pure pans), that one-frame lag is
    exactly what shows up as the crop's focus point visibly drifting
    forward/back instead of tracking smoothly. Making x/y self-contained
    functions of 'on' (like z already is) keeps them in lockstep with z,
    frame for frame.

    x/y are also wrapped in trunc() to force whole-pixel positioning —
    sub-pixel drift between frames is what shows up as a small jitter.
    """
    tf = max(1, total_frames)

    if effect == "zoom_in":
        z = _lerp("1", str(ZOOM_MAX), tf)
        x, y = _centered(z)
    elif effect == "zoom_out":
        z = _lerp(str(ZOOM_MAX), "1", tf)
        x, y = _centered(z)
    elif effect == "pan_left":
        z = str(PAN_ZOOM)
        x, y = _lerp(f"(iw-iw/({z}))", "0", tf), _centered(z)[1]
    elif effect == "pan_right":
        z = str(PAN_ZOOM)
        x, y = _lerp("0", f"(iw-iw/({z}))", tf), _centered(z)[1]
    elif effect == "pan_up":
        z = str(PAN_ZOOM)
        x, y = _centered(z)[0], _lerp(f"(ih-ih/({z}))", "0", tf)
    elif effect == "pan_down":
        z = str(PAN_ZOOM)
        x, y = _centered(z)[0], _lerp("0", f"(ih-ih/({z}))", tf)
    elif effect == "zoom_in_pan_left":
        z = _lerp("1", str(ZOOM_PAN_COMBO_MAX), tf)
        cx, cy = _centered(z)
        x, y = _lerp(f"(iw-iw/({z}))", cx, tf), cy
    elif effect == "zoom_in_pan_right":
        z = _lerp("1", str(ZOOM_PAN_COMBO_MAX), tf)
        cx, cy = _centered(z)
        x, y = _lerp(cx, f"(iw-iw/({z}))", tf), cy
    elif effect in ("fade_in", "fade_out"):
        # gentle zoom under the fade so the frame isn't completely static
        z = _lerp("1", str(ZOOM_MAX), tf)
        x, y = _centered(z)
    else:
        raise ValueError(f"Unknown effect: {effect}")

    return z, f"trunc({x})", f"trunc({y})"


def is_valid_effect(effect: str) -> bool:
    """Check whether *effect* is a known effect name."""
    return effect in set(EFFECTS)
