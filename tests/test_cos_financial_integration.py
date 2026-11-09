"""COS and Fourier must price the *same* financial environment consistently."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hedging import initial_forward_positions, value_book_paths
from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import fixed_eniac_book
from rl_env import prepare_forward_episodes


def _config(tmp_path, method):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    cfg['forward_experiment']['heston']={'kappa':1.,'theta':.04,'v0':.04,'rho':.7,'xi':.2}
    cfg['forward_experiment']['pricing']={'method':method,'n_terms':512,'L':24.} if method=='cos' else {'method':'fourier'}
    path=tmp_path/f'config_{method}.json'
    path.write_text(json.dumps(cfg))
    return path


def test_full_financial_episode_is_cos_consistent(tmp_path):
    config=_config(tmp_path,'cos')
    pool,spec=prepare_forward_episodes(config,n_paths=2,seed=2026)
    marks=pool.book_paths
    assert marks.contract_prices.shape==(2,127,9)
    assert marks.bucket_delta.shape==(2,127,3)
    assert np.all(np.isfinite(marks.contract_prices))
    assert np.all(np.isfinite(marks.bucket_delta))
    # The initial hedge MUST be obtained from the *same* COS deltas as the
    # portfolio mark. Changing marks but leaving the initial hedge on Fourier
    # is an invalid mixed-backend experiment.
    remaining=np.asarray(spec.expiries_days,dtype=float)/spec.trading_days_per_year
    df_dspot=np.exp((pool.market_q.risk_free_rate-pool.market_q.dividend_yield)*remaining)
    wanted=np.clip(-marks.bucket_delta[:,0,:]/df_dspot[None,:],0,spec.max_position)
    np.testing.assert_allclose(pool.initial_positions,wanted,atol=1e-10,rtol=1e-10)


def test_switch_preserves_default_fourier_results(tmp_path):
    cfg=json.loads(_config(tmp_path,'fourier').read_text())
    params=HestonParams(kappa=1.,theta=.04,v0=.04,rho=.7,xi=.2)
    mkt=MarketDynamics(measure='Q',spot0=100.,risk_free_rate=math.log(1.1))
    paths=simulate_heston(params,mkt,n_paths=2,n_steps=126,trading_days_per_year=250,seed=123)
    book=fixed_eniac_book()
    first=value_book_paths(book,paths,params,mkt,trading_days_per_year=250,
                          include_contract_prices=True,include_bucket_deltas=True)
    second=value_book_paths(book,paths,params,mkt,trading_days_per_year=250,
                           include_contract_prices=True,include_bucket_deltas=True,
                           pricing_method='fourier')
    for name in ('value','delta','gamma','variance_sensitivity','contract_prices','bucket_delta'):
        np.testing.assert_array_equal(getattr(first,name),getattr(second,name))


def test_cos_rejects_invalid_configuration_before_expensive_simulation(tmp_path):
    cfg=json.loads(_config(tmp_path,'cos').read_text())
    bads=[{'method':'cos','n_terms':0,'L':24.},
          {'method':'cos','n_terms':512,'L':-1.},
          {'method':'unknown'},
          {'method':'fourier','n_terms':512},
          {'method':'cos','n_terms':512,'L':24.,'unrecognized':True}]
    for i,pricing in enumerate(bads):
        cfg['forward_experiment']['pricing']=pricing
        path=tmp_path/f'bad_{i}.json'
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            prepare_forward_episodes(path,n_paths=1,seed=12)


def test_cos_and_fourier_dataset_configurations_have_different_fingerprints(tmp_path):
    from forward_experiments import _read_config
    fourier=_config(tmp_path,'fourier')
    cos=_config(tmp_path,'cos')
    assert _read_config(fourier)[1]!=_read_config(cos)[1]


def test_creating_cos_campaign_never_overwrites_existing_data(tmp_path):
    from scripts.create_cos_campaign import create_campaign
    root=Path(__file__).resolve().parents[1]
    base_config=tmp_path/'base_config.json'
    base_config.write_text((root/'config.json').read_text())
    base_schedule=tmp_path/'base_schedule.json'
    base_schedule.write_text((root/'configs/expert_schedule.json').read_text())
    out_config=tmp_path/'cos.json'
    out_schedule=tmp_path/'cos_schedule.json'
    dataset_root=f'results/test_cos_campaign_{tmp_path.name}/data'
    result_root=f'results/test_cos_campaign_{tmp_path.name}/models'
    create_campaign(base_config=base_config,base_schedule=base_schedule,
                    out_config=out_config,out_schedule=out_schedule,
                    data_root=dataset_root,result_root=result_root)
    assert json.loads(out_config.read_text())['forward_experiment']['pricing']['method']=='cos'
    assert json.loads(out_schedule.read_text())['result_root']==result_root
    with pytest.raises(FileExistsError):
        create_campaign(base_config=base_config,base_schedule=base_schedule,
                    out_config=out_config,out_schedule=out_schedule,
                    data_root=dataset_root,result_root=result_root)
