"""Independent Fang--Oosterlee COS pricing and convergence tests."""

import math
import numpy as np
import pytest

from heston import HestonParams, MarketDynamics
from pricing import black_scholes_price, heston_greeks_strikes_batch, _adaptive_heston_greeks_near_boundary
from pricing_cos import (
    _log_forward_cumulants,
    heston_cos_greeks_batch,
    heston_cos_greeks_strikes_batch,
)

MARKET = MarketDynamics('Q',100.,math.log(1.1))
STRIKES = np.array([96.,100.,104.])


def hparams(xi):
    return HestonParams(1.,.04,xi,.7,.04)


@pytest.mark.parametrize('T',[1/250,42/250,126/250])
def test_zero_vol_of_vol_matches_exact_black_scholes_for_every_strike(T):
    p=hparams(0.)
    spots=np.array([94.,100.,106.])
    variances=np.array([0.,.04,.15])
    for option_type in ('call','put'):
        cos=heston_cos_greeks_strikes_batch(
            p,MARKET,strikes=STRIKES,maturity=T,spots=spots,
            variances=variances,option_type=option_type,n_terms=256,L=12)
        expected=[]
        for s,v in zip(spots,variances):
            ivar=p.theta*T+(v-p.theta)*(-math.expm1(-p.kappa*T))/p.kappa
            sig=math.sqrt(ivar/T)
            expected.append([black_scholes_price(
                s,float(k),T,sig,MARKET.risk_free_rate,0.,option_type) for k in STRIKES])
        np.testing.assert_allclose(cos.price,expected,atol=1e-9,rtol=0.)


@pytest.mark.parametrize('xi',[0.,.05,.2,.4])
@pytest.mark.parametrize('T',[1/250,42/250,126/250])
def test_cos_vs_current_fourier_in_regular_region(xi,T):
    p=hparams(xi)
    spots=np.array([96.,100.,104.,108.])
    variances=np.array([.015,.04,.08,.12])
    a=heston_cos_greeks_strikes_batch(p,MARKET,strikes=STRIKES,
        maturity=T,spots=spots,variances=variances,n_terms=512,L=14)
    b=heston_greeks_strikes_batch(p,MARKET,strikes=STRIKES,
        maturity=T,spots=spots,variances=variances)
    for field,tolerance in [('price',8e-5),('delta',1e-4),('gamma',1e-4),
                            ('variance_sensitivity',0.08)]:
        np.testing.assert_allclose(getattr(a,field),getattr(b,field),rtol=0.,atol=tolerance)


def test_original_negative_price_state_and_itm_half_price_state():
    p=hparams(.4)
    states=[(94.258753,0.,104.,1/250),(106.24767,.0001084851,96.,1/250)]
    for s,v,k,T in states:
        a=heston_cos_greeks_batch(
            p,MARKET,strike=k,maturity=T,spots=np.array([s]),
            variances=np.array([v]),n_terms=512,L=14)
        b=_adaptive_heston_greeks_near_boundary(
            p,MARKET,spot=s,variance=v,strike=k,maturity=T)
        for field,tolerance in [('price',2e-6),('delta',2e-5),('gamma',5e-5),
                                ('variance_sensitivity',.01)]:
            assert abs(float(getattr(a,field)[0])-float(getattr(b,field)))<tolerance


def test_known_disagreement_with_existing_fast_fourier_uses_independent_check():
    """Fourier fast routine was inaccurate at one low-v far-OTM state.

    Agreement with it cannot be used as a definition of COS accuracy.
    """
    p=hparams(.4)
    s,v,k,T=94.,.0001,100.,.04
    cos=heston_cos_greeks_batch(p,MARKET,strike=k,maturity=T,
        spots=np.array([s]),variances=np.array([v]),n_terms=512,L=24)
    independent=_adaptive_heston_greeks_near_boundary(
        p,MARKET,spot=s,variance=v,strike=k,maturity=T)
    assert cos.price[0]==pytest.approx(independent.price,rel=.001,abs=5e-7)
    assert cos.delta[0]==pytest.approx(independent.delta,rel=.001,abs=1e-7)
    assert cos.variance_sensitivity[0]==pytest.approx(independent.variance_sensitivity,rel=.005,abs=.002)


@pytest.mark.parametrize('xi',[0.,.2,.4])
def test_put_call_parity_and_delta_gamma_parity(xi):
    p=hparams(xi)
    s=np.array([96.,100.,110.]);v=np.array([0.,.04,.07]);T=.168
    common=dict(strikes=STRIKES,maturity=T,spots=s,variances=v,n_terms=512,L=14)
    call=heston_cos_greeks_strikes_batch(p,MARKET,**common)
    put=heston_cos_greeks_strikes_batch(p,MARKET,option_type='put',**common)
    parity=s[:,None]-STRIKES[None,:]*math.exp(-MARKET.risk_free_rate*T)
    np.testing.assert_allclose(call.price-put.price,parity,atol=5e-10)
    np.testing.assert_allclose(call.delta-put.delta,1.,atol=5e-10)
    np.testing.assert_allclose(call.gamma,put.gamma,atol=5e-10)
    np.testing.assert_allclose(call.variance_sensitivity,put.variance_sensitivity,atol=5e-10)


@pytest.mark.parametrize('xi',[.05,.2,.4])
def test_cos_N_and_truncation_L_convergence_in_regular_region(xi):
    p=hparams(xi)
    s=np.array([96.,100.,104.]);v=np.array([.02,.04,.1])
    kw=dict(strikes=STRIKES,maturity=.168,spots=s,variances=v)
    a=heston_cos_greeks_strikes_batch(p,MARKET,n_terms=512,L=12,**kw)
    b=heston_cos_greeks_strikes_batch(p,MARKET,n_terms=1024,L=12,**kw)
    c=heston_cos_greeks_strikes_batch(p,MARKET,n_terms=1024,L=18,**kw)
    assert np.max(np.abs(a.price-b.price))<2e-5
    assert np.max(np.abs(b.price-c.price))<2e-5
    assert np.max(np.abs(a.delta-b.delta))<2e-5
    assert np.max(np.abs(b.delta-c.delta))<2e-5


def test_cos_cumulants_match_constant_variance_limit():
    p=hparams(0.)
    variances=np.array([0.,.04,.2])
    T=.1
    c1,c2=_log_forward_cumulants(p,T,variances)
    A=p.theta*T+(variances-p.theta)*(-math.expm1(-T))
    np.testing.assert_allclose(c1,-A/2,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(c2,A,rtol=1e-12,atol=1e-14)


def test_cos_rejects_invalid_measure_and_inputs():
    p=hparams(.2)
    k=dict(strikes=STRIKES,maturity=.1,spots=np.array([100.]),variances=np.array([.04]))
    with pytest.raises(ValueError,match='Q dynamics'):
        heston_cos_greeks_strikes_batch(p,MarketDynamics('P',100.,.05,physical_total_return=.07),**k)
    with pytest.raises(ValueError,match='strikes'):
        heston_cos_greeks_strikes_batch(p,MARKET,**{**k,'strikes':np.array([0.])})
    with pytest.raises(ValueError,match='n_terms'):
        heston_cos_greeks_strikes_batch(p,MARKET,**k,n_terms=4)
    with pytest.raises(ValueError,match='L'):
        heston_cos_greeks_strikes_batch(p,MARKET,**k,L=-1)
    with pytest.raises(ValueError,match='maturity'):
        heston_cos_greeks_strikes_batch(p,MARKET,**{**k,'maturity':0.})
    with pytest.raises(ValueError,match='invalid spots'):
        heston_cos_greeks_strikes_batch(p,MARKET,**{**k,'variances':np.array([-1.])})


def test_cos_analytic_spot_greeks_and_variance_sensitivity_match_finite_differences():
    p=hparams(.2);s=100.;v=.04;k=100.;T=.168
    def price(s,v):
        return heston_cos_greeks_batch(p,MARKET,strike=k,maturity=T,
            spots=np.array([s]),variances=np.array([v]),n_terms=512,L=24).price[0]
    out=heston_cos_greeks_batch(p,MARKET,strike=k,maturity=T,
            spots=np.array([s]),variances=np.array([v]),n_terms=512,L=24)
    h=.02
    fd_delta=(price(s+h,v)-price(s-h,v))/(2*h)
    fd_gamma=(price(s+h,v)-2*price(s,v)+price(s-h,v))/h**2
    dv=1e-4
    fd_variance=(price(s,v+dv)-price(s,v-dv))/(2*dv)
    assert float(out.delta[0])==pytest.approx(fd_delta,abs=2e-6)
    assert float(out.gamma[0])==pytest.approx(fd_gamma,abs=1e-5)
    assert float(out.variance_sensitivity[0])==pytest.approx(fd_variance,rel=1e-4)


def test_benchmark_cli_writes_csv_and_json_without_modifying_financial_engine(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'benchmark.csv'
    cp = subprocess.run(
        [sys.executable, str(root / 'scripts/benchmark_heston_cos.py'),
         '--n-states', '3', '--terms', '256', '--xis', '0',
         '--maturities', '.168', '--csv', str(out)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert out.is_file(), cp.stdout + cp.stderr
    assert out.with_suffix('.json').is_file()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]['status'] == 'ok'
    assert abs(float(rows[0]['max_abs_price'])) < 1e-7


def test_benchmark_cli_rejects_held_out_test_split(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cp = subprocess.run(
        [sys.executable, str(root / 'scripts/benchmark_heston_cos.py'),
         '--npz-file', str(tmp_path / 'test.npz')],
        cwd=root, capture_output=True, text=True,
    )
    assert cp.returncode != 0
    assert 'held-out' in cp.stderr
