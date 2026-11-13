"""Regression coverage for fair Heston/Black/periodic forward hedging baselines."""
import json
from pathlib import Path

import numpy as np
import pytest

from financial_baselines import (
    DEFAULT_STRATEGIES, STRATEGIES, _actions_to_target, _bucket_deltas,
    _position_targets, black_call_delta, black_call_price,
    evaluate_financial_baseline, invert_black_call_iv,
)
from forward_experiments import evaluate_baseline
from hedging import BookPaths
from heston import HestonParams, HestonPaths, MarketDynamics
from rl_env import ForwardEnvConfig, ForwardEpisodePool, prepare_forward_episodes


def _dummy():
    n, t = 2, 127
    spot = np.full((n,t),100.)
    spots=np.array([100.,105.])
    spot[:]=spots[:,None]
    spot[:,1:]+=np.arange(126)[None,:]*.1
    v=np.full((n,t),.04)
    paths=HestonPaths(np.arange(t)/250,spot,v,v.copy(),'Q')
    bd=np.zeros((n,t,3),dtype=float)
    bd[:, :42, 0]=-1.5
    bd[:, :84, 1]=-1.0
    bd[:, :126, 2]=-2.0
    bp=BookPaths(np.zeros((n,t)),bd.sum(axis=2),np.zeros((n,t)),
                 np.zeros((n,t)),np.zeros((n,t)),
                 np.ones((n,t,9)),bd)
    cfg=ForwardEnvConfig()
    initial=np.array([[1.5,1.,2.],[1.5,1.,2.]])
    return ForwardEpisodePool(paths,bp,initial,MarketDynamics('Q',100.,.09531017980432493)),cfg


def test_black_call_delta_known_values_and_zero_vol():
    assert black_call_delta(np.array(100.),np.array(100.),1.,risk_free_rate=0.,
                            dividend_yield=0.,sigma=.2) == pytest.approx(.5398278373)
    out=black_call_delta(np.array([120.,80.,100.]),np.array(100.),1.,
                         risk_free_rate=0.,dividend_yield=0.,sigma=0.)
    np.testing.assert_array_equal(out,[1.,0.,.5])
    with pytest.raises(ValueError):
        black_call_delta(np.array([100.]),np.array([100.]),1.,risk_free_rate=0.,
                         dividend_yield=0.,sigma=-1.)


def test_iv_reconstruction_and_delta_not_equal_heston_in_general():
    spots=np.array([[85.,100.,115.],[85.,100.,115.]])
    k=np.array([[95.,105.,100.]])
    sigma=np.array([[.15,.25,.32],[.45,.21,.29]])
    price=black_call_price(spots,k,.4,risk_free_rate=.03,
                           dividend_yield=.01,sigma=sigma)
    inferred=invert_black_call_iv(spots,k,.4,market_rate=.03,
                                  dividend_yield=.01,call_prices=price)
    np.testing.assert_allclose(inferred,sigma,atol=1e-7)
    reconstructed=black_call_price(spots,k,.4,risk_free_rate=.03,
                                   dividend_yield=.01,sigma=inferred)
    np.testing.assert_allclose(price,reconstructed,atol=1e-8)
    with pytest.raises(ValueError, match='no-arbitrage'):
        invert_black_call_iv(np.array([100.]),np.array([100.]),1.,
                             market_rate=0.,dividend_yield=0.,call_prices=np.array([-1.]))


def test_heston_original_baselines_identical():
    pool,cfg=_dummy()
    for old,new in [('no_hedge','no_hedge'),('hold_initial','hold_initial'),
                    ('delta_daily','heston_daily'),
                    ('delta_periodic_21','heston_periodic_21'),
                    ('delta_band_0p5','heston_band_0p5')]:
        orig=evaluate_baseline(pool,cfg,old)
        result=evaluate_financial_baseline(pool,cfg,new)
        np.testing.assert_array_equal(result.terminal_pnl,orig.terminal_pnl)
        np.testing.assert_array_equal(result.transaction_cost,orig.transaction_cost)
        np.testing.assert_array_equal(result.position,orig.position)


def test_periodic_trades_only_at_21_day_marks_and_respects_execution():
    pool,cfg=_dummy()
    delta=_bucket_deltas(pool,cfg,source='black20')
    targets=_position_targets(delta,pool,cfg)
    initial,actions=_actions_to_target(pool,cfg,targets,frequency=21)
    assert initial.shape == (2,3)
    assert np.any(initial>0)
    assert np.all(actions[:,np.arange(126)%21!=0,:] == 0)
    assert np.max(np.abs(actions)) <= cfg.max_trade_increment
    got=evaluate_financial_baseline(pool,cfg,'black20_periodic_21')
    daily=evaluate_financial_baseline(pool,cfg,'black20_daily')
    assert np.isfinite(got.terminal_pnl).all()
    assert np.isfinite(daily.terminal_pnl).all()
    assert np.all(got.position[:,-1,:]==0)
    assert np.all(got.initial_cost>0)
    assert np.all(got.transaction_cost>=got.initial_cost)


def test_minimum_variance_requires_explicit_heston_params():
    pool,cfg=_dummy()
    with pytest.raises(ValueError,match='Heston parameters'):
        _bucket_deltas(pool,cfg,source='heston_mv')


def test_constant_variance_mv_equals_heston_baseline(tmp_path):
    src=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    src['forward_experiment']['heston']={'kappa':1.,'theta':.04,'xi':0.,'rho':.7,'v0':.04}
    src['forward_experiment']['pricing']={'method':'cos','n_terms':512,'L':24.}
    config=tmp_path/'config.json'
    config.write_text(json.dumps(src))
    pool,cfg=prepare_forward_episodes(config,n_paths=2,seed=4)
    h=HestonParams(**src['forward_experiment']['heston'])
    delta=_bucket_deltas(pool,cfg,source='heston_mv',heston=h)
    np.testing.assert_allclose(delta,pool.book_paths.bucket_delta,atol=1e-8)
    baseline=evaluate_financial_baseline(pool,cfg,'heston_daily')
    mv=evaluate_financial_baseline(pool,cfg,'heston_mv_daily',heston=h)
    black=evaluate_financial_baseline(pool,cfg,'black20_daily')
    np.testing.assert_allclose(baseline.terminal_pnl,mv.terminal_pnl,atol=1e-8)
    # With xi=0 and v0=theta=.04, Heston is exactly Black at sigma=20%.
    np.testing.assert_allclose(baseline.terminal_pnl,black.terminal_pnl,atol=1e-8)


def test_both_new_baselines_have_different_initial_targets():
    pool,cfg=_dummy()
    black=evaluate_financial_baseline(pool,cfg,'black20_daily')
    heston=evaluate_financial_baseline(pool,cfg,'heston_daily')
    assert not np.allclose(black.position[:,0,:],heston.position[:,0,:])
    assert not np.allclose(black.initial_cost,heston.initial_cost)
    assert 'black20_periodic_21' in DEFAULT_STRATEGIES
    assert 'black_iv_daily' in STRATEGIES and 'black_iv_daily' not in DEFAULT_STRATEGIES
