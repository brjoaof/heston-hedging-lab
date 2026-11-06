"""Pure infrastructure tests; SB3/Gym smoke is a separate machine-side gate."""
from argparse import Namespace
import json
from pathlib import Path

import pytest
import numpy as np

from forward_experiments import iter_train_chunks
from train_sac_scale import (
    SELECTION_METRIC, ScaleProgress, assert_same_plan, code_fingerprint,
    code_version_compatible,
    q95_is_better, scale_plan, worker_stream_seed,
    _load_recovery, _run_sha, _save_recovery,
)
from rl_env import ForwardEnvConfig


ROOT = Path(__file__).resolve().parents[1]


def make_args(**changes):
    data = dict(timesteps=50_000_000, n_envs=30, eval_every=2_500_000,
                resume_every=250_000, progress_every=50_000,
                train_chunk_paths=16, learning_starts=50_000,
                batch_size=256, buffer_size=500_000,
                seed=123, train_seed=551177,
                vec_env='subproc', gamma=0.995, tau=0.002,
                learning_rate=1e-4, penalty=2.0,
                action_mode='band', max_band_width=1.5,
                max_trade_increment=1.0, regime_xi=0.2,
                config=str(ROOT/'config.json'), splits_dir=str(ROOT/'results/splits'))
    data.update(changes)
    return Namespace(**data)


def test_q95_selects_smaller_risk_not_larger_reward():
    low_q95 = dict(q95_loss=40., mean_episode_reward=-5000.)
    high_q95 = dict(q95_loss=80., mean_episode_reward=20.)
    assert q95_is_better(low_q95, float('inf'))
    assert not q95_is_better(high_q95, 40.)
    assert not q95_is_better(dict(q95_loss=40.), 40.)
    with pytest.raises(ValueError):
        q95_is_better(dict(q95_loss=float('nan')), 80.)


def test_defaults_follow_eniac_and_vector_rounding():
    from forward_experiments import _read_config
    from rl_env import ForwardEnvConfig
    args=make_args()
    cfg=ForwardEnvConfig.from_mapping(_read_config(args.config)[0])
    plan=scale_plan(args,cfg=cfg,config_sha='finance',val_sha='validation')
    assert plan['selected_by']==SELECTION_METRIC
    assert plan['band_execution']=='delta_center'
    assert plan['n_envs']==30
    assert plan['requested_timesteps']==50_000_000
    assert plan['timesteps']==50_000_010  # SB3 batches 30 parallel transitions
    assert plan['learning_starts']==50_000
    assert plan['buffer_size']==500_000
    assert plan['config']['max_trade_increment']==3.0
    assert plan['penalty']==2.0
    assert plan['train_freq_steps'] == 1
    assert plan['gradient_steps'] == 1
    assert plan['torch_num_threads'] == 1


def test_trade_has_original_increment_limit():
    from forward_experiments import _read_config
    args=make_args(action_mode='trade')
    cfg=ForwardEnvConfig.from_mapping(_read_config(args.config)[0])
    plan=scale_plan(args,cfg=cfg,config_sha='a',val_sha='b')
    assert plan['max_trade_increment']==1.0
    assert 'band_execution' not in plan


@pytest.mark.parametrize('changed', [
    {'seed':124}, {'timesteps':20_000_000}, {'action_mode':'trade'},
    {'penalty':0.03}, {'vec_env':'dummy'}, {'n_envs':2},
])
def test_resume_rejects_scientific_configuration_change(changed):
    from forward_experiments import _read_config
    cfg=ForwardEnvConfig.from_mapping(_read_config(ROOT/'config.json')[0])
    old=scale_plan(make_args(),cfg=cfg,config_sha='a',val_sha='b')
    args=make_args(**changed)
    new=scale_plan(args,cfg=cfg,config_sha='a',val_sha='b')
    with pytest.raises(ValueError,match='Resume configuration mismatch'):
        assert_same_plan(old,new)


def test_code_fingerprint_and_worker_streams():
    assert len(code_fingerprint())==64
    values=[worker_stream_seed(551177,i) for i in range(30)]
    assert len(set(values))==30
    assert worker_stream_seed(551177,1)==worker_stream_seed(551177,1)
    with pytest.raises(ValueError):worker_stream_seed(-1,1)


def test_chunk_resume_at_exact_index():
    cfg=ROOT/'config.json'
    a=list(iter_train_chunks(cfg,base_seed=17171,chunk_size=1,n_chunks=3,start_chunk=2))
    b=list(iter_train_chunks(cfg,base_seed=17171,chunk_size=1,n_chunks=3))[2]
    assert len(a)==1 and a[0][0]==b[0]
    np.testing.assert_array_equal(a[0][1].paths.spot,b[1].paths.spot)
    with pytest.raises(ValueError):
        list(iter_train_chunks(cfg,base_seed=1,chunk_size=1,n_chunks=2,start_chunk=3))


def test_resumed_progress_counts_from_previous_steps(monkeypatch):
    ticks=iter([100.,120.])
    monkeypatch.setattr('train_sac_scale.time.perf_counter', lambda: next(ticks))
    progress=ScaleProgress(1000,100,completed=500,previous_wall=30.)
    rec=progress.snapshot(600)
    assert rec['elapsed_seconds']==50.
    assert rec['eta_seconds']==80.
    assert rec['progress_percent']==60.


def test_recovery_artifact_round_trip_with_stubs(tmp_path):
    class Model:
        def save(self,path):Path(str(path)+'.zip').write_text('model')
        def save_replay_buffer(self,path):Path(path).write_text('replay')
    class Base:
        def env_method(self,name):
            assert name=='export_training_state'
            return [{'seed':777}]
    class Vec:
        venv=Base()
        def save(self,path):Path(path).write_text('norm')
    class Torch:
        @staticmethod
        def get_rng_state():return np.zeros(8,dtype=np.uint8)
    out=tmp_path/'run';out.mkdir()
    sha='abc'
    root=_save_recovery(Model(),Vec(),out,step=300,
                        callback_state={'best_q95_loss':35,'n_evaluations':3},
                        run_sha=sha,torch_module=Torch())
    loaded, meta=_load_recovery(out,sha)
    assert loaded==root
    assert meta['timesteps']==300
    assert (root/'replay_buffer.pkl').read_text()=='replay'
    with pytest.raises(ValueError, match='do not match'):
        _load_recovery(out,'wrong_sha')


def test_json_round_trip_resume_plan_with_original_source_fingerprint():
    """Reproduce the first real-world smoke resume failure (tuples vs lists).

    A pre-fix run.json has tuple fields serialized into JSON lists and holds
    the original code fingerprint. Only the audited source migration is legal.
    """
    from forward_experiments import _read_config
    cfg = ForwardEnvConfig.from_mapping(_read_config(ROOT/'config.json')[0])
    requested = scale_plan(make_args(action_mode='trade',n_envs=2,timesteps=6000,
                                     learning_starts=256,buffer_size=5000,
                                     eval_every=1500,resume_every=1000,progress_every=500,
                                     vec_env='dummy'),cfg=cfg,config_sha='cfg',val_sha='val')
    old = json.loads(json.dumps(requested))
    assert isinstance(old['config']['expiries_days'], list)
    assert isinstance(requested['config']['expiries_days'], tuple)
    compat = json.loads((ROOT/'scale_resume_compat.json').read_text())
    assert compat['schema_version'] == 2
    # The older compatibility entries cover ONLY the historical source
    # fingerprint. A new numerical pricing backend must fail closed on old
    # SAC recovery snapshots: do not silently change an ongoing experiment.
    for previous in compat['compatible_previous_sha256']:
        assert code_version_compatible(previous, compat['fixed_code_sha256'])
        assert not code_version_compatible(previous, requested['code_sha256'])
    assert_same_plan(old, requested)  # tuples -> JSON arrays; source unchanged
    assert not code_version_compatible('f'*64, requested['code_sha256'])
    assert not code_version_compatible(compat['compatible_previous_sha256'][0], '0'*64)


@pytest.mark.parametrize('field,new_value',[
    ('n_envs', 4), ('penalty', 0.5), ('validation_data_sha256','other'),
    ('financial_config_sha256','other'), ('seed', 999),
])
def test_original_run_cannot_resume_with_changed_scientific_controls(field,new_value):
    from forward_experiments import _read_config
    cfg = ForwardEnvConfig.from_mapping(_read_config(ROOT/'config.json')[0])
    req = scale_plan(make_args(),cfg=cfg,config_sha='cfg',val_sha='val')
    old = json.loads(json.dumps(req))
    old['code_sha256'] = req['code_sha256']
    req[field] = new_value
    with pytest.raises(ValueError,match='Resume configuration mismatch'):
        assert_same_plan(old, req)


def test_legacy_resume_rejects_changed_financial_config_inner_field():
    from forward_experiments import _read_config
    cfg = ForwardEnvConfig.from_mapping(_read_config(ROOT/'config.json')[0])
    req = scale_plan(make_args(),cfg=cfg,config_sha='cfg',val_sha='val')
    old = json.loads(json.dumps(req))
    old['code_sha256'] = req['code_sha256']
    req['config']['fixed_cost'] = 999
    with pytest.raises(ValueError,match='Resume configuration mismatch for config'):
        assert_same_plan(old,req)


def test_resumed_stream_discards_inflight_episode_but_preserves_next_path(monkeypatch):
    """Regression: old hook returned terminal observation on later auto-reset.

    Restore a pre-fix in-flight episode at day 114, force a fresh reset,
    and check that no terminal episode can be stepped accidentally.  This
    small test needs no Gymnasium, SAC or user-generated pickle files.
    """
    from types import SimpleNamespace
    from train_sac_scale import _ResumableStream, SCHEMA_VERSION

    class FakeGymBase:
        def reset(self, *, seed=None, options=None):
            self.reset_count += 1
            self.core.day = 0
            self.core.path_index = self._order[self._cursor]
            self._cursor += 1
            return (np.asarray([self.core.path_index], dtype=float), {})

    class Stream(_ResumableStream, FakeGymBase):
        def __init__(self):
            self.reset_count = 0
            self._order = np.array([0, 1, 2])
            self._cursor = 0
            self.chunk_index = 0
            self.generated_chunks = 1
            self.core = SimpleNamespace(config='same', path_index=None,
                                        day=0, position=None, cash=None)
            self.initialize_scale('unused.json', 19, 3, 5)

    s = Stream()
    state = {
        'schema_version': SCHEMA_VERSION,
        'seed': 19, 'n_chunks': 5, 'paths_per_chunk': 3,
        'core': SimpleNamespace(config='same', path_index=0, day=114,
                                position=np.array([[1.]]), cash=np.array([0.])),
        'order': np.array([0, 1, 2]), 'cursor': 1,
        'chunk_index': 0, 'generated_chunks': 1,
    }
    monkeypatch.setattr('train_sac_scale.iter_train_chunks', lambda *a, **kw: iter(()))
    s.import_training_state(state)
    assert s.core.path_index is None
    assert s.core.position is None and s.core.cash is None
    obs, _ = s.reset()
    assert s.reset_count == 1
    assert s.core.day == 0
    assert s.core.path_index == 1  # next UNUSED Q trajectory, not replayed path 0
    assert s._cursor == 2
    np.testing.assert_array_equal(obs, [1.])
    # When DummyVecEnv automatically resets after a terminal transition, the
    # same ordinary reset() must create a fresh episode, never a terminal obs.
    s.core.day = 126
    obs2, _ = s.reset()
    assert s.reset_count == 2 and s.core.day == 0
    assert s.core.path_index == 2
    np.testing.assert_array_equal(obs2, [2.])


def test_loading_checkpoint_requests_fresh_sb3_reset():
    """Guard against accidentally removing force_reset=True on future edits."""
    import inspect
    from train_sac_scale import train
    src = inspect.getsource(train)
    assert 'force_reset=True' in src
    assert 'if model._last_obs is not None:' in src


def test_legacy_resume_manifest_includes_uploaded_checkpoint():
    from train_sac_scale import code_fingerprint
    compat = json.loads((ROOT/'scale_resume_compat.json').read_text())
    assert compat['fixed_code_sha256'] != code_fingerprint()
    assert not code_version_compatible(compat['compatible_previous_sha256'][0], code_fingerprint())
    assert '75ca78ee958d4f3326867c7cc59de1c87422dc6f267768c06e48bd0b25c8ce58' in compat['compatible_previous_sha256']
    assert 'ba32bff3d9ed80a47cb4afa7540c7811ead9a5ab0881262a3e9bd4ff358f89e8' in compat['compatible_previous_sha256']


@pytest.mark.skipif(__import__('importlib').util.find_spec('gymnasium') is None or
                    __import__('importlib').util.find_spec('stable_baselines3') is None,
                    reason='Requires the real Gymnasium and Stable-Baselines3 VecEnv')
def test_real_vecenv_automatically_resets_after_resumed_episode():
    """Exercise the exact DummyVecEnv auto-reset path that crashed at 3k."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train_sac_scale import ScaleTradeStream

    args = dict(config_path=str(ROOT/'config.json'), train_seed=981271,
                paths_per_chunk=2, n_chunks=4, penalty=2.0,
                max_trade_increment=1.0, include_bucket_deltas=True)
    first = ScaleTradeStream(**args)
    try:
        first.reset()
        for _ in range(114):
            _, _, ended, _, _ = first.step(np.zeros(3, dtype=np.float32))
            assert not ended
        assert first.core.day == 114
        saved = first.export_training_state()
    finally:
        first.close()

    restored = ScaleTradeStream(**args)
    restored.import_training_state(saved)
    assert restored.core.path_index is None
    vec = DummyVecEnv([lambda: restored])
    try:
        vec.reset()  # force-reset behavior used by SAC.load(force_reset=True)
        assert restored.core.day == 0
        for _ in range(260):  # crosses TWO automatic episode boundaries
            vec.step(np.zeros((1, 3), dtype=np.float32))
        assert restored.generated_chunks >= 2
    finally:
        vec.close()


def test_eniac_cpu_thread_cap_and_subproc_path_are_explicit():
    """Prevent silent drift from the 30-env CPU training protocol."""
    import inspect
    from types import SimpleNamespace
    from train_sac_scale import set_eniac_cpu_threads, train
    calls = []
    set_eniac_cpu_threads(SimpleNamespace(set_num_threads=calls.append))
    assert calls == [1]
    src = inspect.getsource(train)
    assert 'SubprocVecEnv(factories, start_method="forkserver")' in src
    assert 'train_freq=(1, "step"), gradient_steps=1' in src


def test_delta_center_is_frozen_for_eniac_band_specialists():
    """Do not accidentally reintroduce the uploaded Band-Boundary variant."""
    from band_policy import BAND_EXECUTION, band_to_increment
    assert BAND_EXECUTION == 'delta_center'
    pos = np.array([0.0, 1.0, 2.0])
    target = np.array([1.0, 1.1, 1.0])
    width = np.array([0.3, 0.3, 0.3])
    delta = band_to_increment(pos, target, width, max_position=3.)
    np.testing.assert_array_equal(delta, [1.0, 0.0, -1.0])


def test_git_ready_ignores_experimental_results_but_keeps_reproducible_config():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / '.gitignore').read_text()
    assert '/results/' in gitignore
    assert '.venv/' in gitignore
    assert '*.zip' in gitignore
    assert (root / 'scripts' / 'setup_gitignore.py').is_file()
    assert (root / '.gitattributes').is_file()
    assert (root / 'config.json').is_file()
    assert (root / 'docs' / 'GIT_SERVER.md').is_file()


@pytest.mark.skipif(__import__('importlib').util.find_spec('gymnasium') is None or
                    __import__('importlib').util.find_spec('stable_baselines3') is None,
                    reason='Requires real Gymnasium / Stable-Baselines3')
def test_real_subproc_env_stream_state_round_trip():
    """Small subprocess integration test; the scale server will use 30 workers."""
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from train_sac_scale import make_worker

    opts = dict(action_mode='trade', config_path=str(ROOT/'config.json'),
                train_seed=921, paths_per_chunk=2, n_chunks=5,
                penalty=2.0, max_trade_increment=1.0, max_band_width=1.5)
    fns = [partial(make_worker, i, **opts) for i in range(2)]
    env = SubprocVecEnv(fns, start_method='forkserver')
    try:
        obs = env.reset()
        assert obs.shape == (2, 20)
        next_obs, _, _, _ = env.step(np.zeros((2, 3), dtype=np.float32))
        assert next_obs.shape == (2, 20)
        states = env.env_method('export_training_state')
        assert len(states) == 2
        assert states[0]['seed'] != states[1]['seed']
        for i, state in enumerate(states):
            env.env_method('import_training_state', state, indices=[i])
        obs_after = env.reset()
        assert obs_after.shape == (2, 20)
        env.step(np.zeros((2, 3), dtype=np.float32))
    finally:
        env.close()


def test_gitignore_setup_is_idempotent_and_does_not_overwrite_rules(tmp_path):
    from scripts.setup_gitignore import update_gitignore
    existing = tmp_path / '.gitignore'
    existing.write_text('important_local_pattern/\n', encoding='utf-8')
    assert update_gitignore(existing)
    assert not update_gitignore(existing)
    text = existing.read_text(encoding='utf-8')
    assert 'important_local_pattern/' in text
    assert '/results/' in text
    assert '*.zip' in text
