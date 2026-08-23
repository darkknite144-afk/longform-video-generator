"""
Tests for longform.effects — effect assignment and zoompan expressions.
"""

import pytest

from longform.config import EFFECTS, PAN_ZOOM, ZOOM_MAX, ZOOM_PAN_COMBO_MAX
from longform.effects import assign_effects, is_valid_effect, zoompan_expr


class TestAssignEffects:
    """Test the assign_effects function."""

    def test_correct_count(self):
        effects = assign_effects(10)
        assert len(effects) == 10

    def test_zero_scenes(self):
        effects = assign_effects(0)
        assert effects == []

    def test_single_scene(self):
        effects = assign_effects(1)
        assert len(effects) == 1
        assert effects[0] in EFFECTS

    def test_more_than_pool(self):
        effects = assign_effects(len(EFFECTS) + 5)
        assert len(effects) == len(EFFECTS) + 5

    def test_all_valid(self):
        effects = assign_effects(50)
        for e in effects:
            assert e in EFFECTS

    def test_deterministic_seeded(self):
        """Same n_scenes should produce identical output (seeded RNG)."""
        e1 = assign_effects(20)
        e2 = assign_effects(20)
        assert e1 == e2

    def test_no_back_to_back_repeat(self):
        """No two consecutive effects should be the same."""
        effects = assign_effects(100)
        for i in range(1, len(effects)):
            assert effects[i] != effects[i - 1], (
                f"Back-to-back repeat at index {i}: {effects[i]}"
            )

    def test_large_count(self):
        effects = assign_effects(200)
        assert len(effects) == 200
        for e in effects:
            assert e in EFFECTS

    def test_exact_pool_size(self):
        effects = assign_effects(len(EFFECTS))
        assert len(effects) == len(EFFECTS)
        assert set(effects) == set(EFFECTS)


class TestZoompanExpr:
    """Test the zoompan_expr function."""

    def test_zoom_in_returns_three_strings(self):
        z, x, y = zoompan_expr("zoom_in", 100)
        assert isinstance(z, str)
        assert isinstance(x, str)
        assert isinstance(y, str)

    def test_zoom_out(self):
        z, x, y = zoompan_expr("zoom_out", 100)
        assert "trunc" in x
        assert "trunc" in y

    def test_pan_left(self):
        z, x, y = zoompan_expr("pan_left", 100)
        assert z == str(PAN_ZOOM)
        assert "trunc" in x

    def test_pan_right(self):
        z, x, y = zoompan_expr("pan_right", 100)
        assert z == str(PAN_ZOOM)

    def test_pan_up(self):
        z, x, y = zoompan_expr("pan_up", 100)
        assert z == str(PAN_ZOOM)

    def test_pan_down(self):
        z, x, y = zoompan_expr("pan_down", 100)
        assert z == str(PAN_ZOOM)

    def test_zoom_in_pan_left(self):
        z, x, y = zoompan_expr("zoom_in_pan_left", 100)
        assert "on" in z

    def test_zoom_in_pan_right(self):
        z, x, y = zoompan_expr("zoom_in_pan_right", 100)
        assert "on" in z

    def test_fade_in(self):
        z, x, y = zoompan_expr("fade_in", 100)
        assert "on" in z
        assert "trunc" in x

    def test_fade_out(self):
        z, x, y = zoompan_expr("fade_out", 100)
        assert "on" in z

    def test_unknown_effect_raises(self):
        with pytest.raises(ValueError, match="Unknown effect"):
            zoompan_expr("nonexistent_effect", 100)

    def test_zero_frames_does_not_crash(self):
        """total_frames=0 should be clamped to max(1, 0) = 1."""
        z, x, y = zoompan_expr("zoom_in", 0)
        assert isinstance(z, str)

    def test_negative_frames_does_not_crash(self):
        z, x, y = zoompan_expr("zoom_in", -5)
        assert isinstance(z, str)

    def test_x_y_have_trunc(self):
        """All effects should wrap x/y in trunc() for whole-pixel positioning."""
        for effect in EFFECTS:
            z, x, y = zoompan_expr(effect, 100)
            assert "trunc" in x, f"x missing trunc for {effect}"
            assert "trunc" in y, f"y missing trunc for {effect}"

    def test_z_contains_on_for_changing_zoom(self):
        """Effects with changing zoom should reference 'on' in z expression."""
        changing = ["zoom_in", "zoom_out", "zoom_in_pan_left",
                    "zoom_in_pan_right", "fade_in", "fade_out"]
        for effect in changing:
            z, _, _ = zoompan_expr(effect, 100)
            assert "on" in z, f"z missing 'on' for {effect}"

    def test_pure_pan_z_is_constant(self):
        """Pure pan effects should have a constant z (no 'on')."""
        pure_pans = ["pan_left", "pan_right", "pan_up", "pan_down"]
        for effect in pure_pans:
            z, _, _ = zoompan_expr(effect, 100)
            assert "on" not in z, f"z should be constant for {effect}"

    def test_all_effects_produce_valid_output(self):
        """Every effect in EFFECTS should produce non-empty z, x, y."""
        for effect in EFFECTS:
            z, x, y = zoompan_expr(effect, 90)
            assert len(z) > 0
            assert len(x) > 0
            assert len(y) > 0

    def test_large_frame_count(self):
        z, x, y = zoompan_expr("zoom_in", 10000)
        assert "on/10000" in z

    def test_single_frame(self):
        z, x, y = zoompan_expr("zoom_in", 1)
        assert "on/1" in z


class TestIsValidEffect:
    """Test the is_valid_effect utility."""

    def test_valid_effects(self):
        for effect in EFFECTS:
            assert is_valid_effect(effect) is True

    def test_invalid_effect(self):
        assert is_valid_effect("nonexistent") is False

    def test_empty_string(self):
        assert is_valid_effect("") is False

    def test_case_sensitive(self):
        assert is_valid_effect("ZOOM_IN") is False
        assert is_valid_effect("Zoom_In") is False
