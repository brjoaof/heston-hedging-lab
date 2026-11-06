"""Deterministic ETA tests without Gymnasium or Stable-Baselines3."""
import json

import pytest

from train_sac_trade import TrainingProgress, _format_duration, _progress_line


def test_time_display_handles_days_unknown_and_zero():
    assert _format_duration(None) == '--'
    assert _format_duration(0) == '00:00:00'
    assert _format_duration(3661) == '01:01:01'
    assert _format_duration(2 * 86400 + 72) == '2d 00:01:12'


def test_progress_uses_real_walltime_and_rolls_up_evaluation_overhead():
    now = [100.0]
    progress = TrainingProgress(1000, 250, clock=lambda: now[0])
    assert progress.snapshot(0) is None
    assert progress.snapshot(249) is None
    now[0] = 110.0
    a = progress.snapshot(250)
    assert a['progress_percent'] == 25.0
    assert a['elapsed_seconds'] == 10.0
    assert a['average_steps_per_second'] == pytest.approx(25.0)
    assert a['eta_seconds'] == pytest.approx(30.0)
    assert a['estimated_finish_utc'] is not None
    assert progress.snapshot(250, force=True) is None
    # Simulate a slower second segment including a validation+checkpoint pause.
    now[0] = 130.0
    b = progress.snapshot(500)
    assert b['average_steps_per_second'] == pytest.approx(500/30)
    assert b['rolling_steps_per_second'] == pytest.approx(.65*25 + .35*12.5)
    assert b['eta_seconds'] == pytest.approx(500 / (.65*25 + .35*12.5))
    assert '500/1,000' in _progress_line(b)
    assert 'ETA ~' in _progress_line(b)
    json.dumps(b)
    now[0] = 140.0
    done = progress.snapshot(1000, force=True)
    assert done['eta_seconds'] == 0
    assert done['progress_percent'] == 100.0
    assert progress.snapshot(1000, force=True) is None


def test_progress_forced_end_with_nondivisible_total():
    now = [0.0]
    progress = TrainingProgress(112, 50, clock=lambda: now[0])
    now[0] = 1.0
    assert progress.snapshot(50) is not None
    now[0] = 2.0
    assert progress.snapshot(100) is not None
    now[0] = 3.0
    assert progress.snapshot(112) is None
    assert progress.snapshot(112, force=True)['eta_seconds'] == 0.0


@pytest.mark.parametrize('total,every', [(0,1),(1,0),(-1,5),(100,-1)])
def test_progress_rejects_invalid_intervals(total, every):
    with pytest.raises(ValueError):
        TrainingProgress(total,every)


def test_progress_rejects_invalid_step_counts():
    progress=TrainingProgress(100,10)
    with pytest.raises(ValueError):
        progress.snapshot(-1)
    with pytest.raises(ValueError):
        progress.snapshot(101)
