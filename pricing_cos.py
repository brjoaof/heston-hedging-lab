"""Independent Fourier-cosine (COS) pricer for European options under Heston-Q.

Fang & Oosterlee (2008), SIAM J. Scientific Computing 31(2), 826-848,
DOI: 10.1137/080718061. This is an experimental, standalone implementation;
it does NOT replace pricing.py or any simulator/RL training path.

The COS expansion reconstructs the density of X=log(S_T/F_t) over [a,b].
Using this *strike-independent* variable shares characteristic-function
coefficients across several strikes. Analytic payoff integrals (chi, psi)
then give OTM call/put prices, with put-call parity for the ITM option.

Truncation: [c1 - L sqrt(c2), c1 + L sqrt(c2)] using the exact conditional
first and second cumulants of X. This is intentionally a documented design
choice, not the c1,c2,c4-based interval suggested in Fang--Oosterlee; users
MUST check convergence in both N and L before using scientific results.

The option Greek dC/dv is the derivative w.r.t. current instant variance,
NOT the implied-volatility vega. Delta/Gamma hold variance fixed.
"""

from __future__ import annotations

import math

import numpy as np

from heston import HestonParams, MarketDynamics
from pricing import OptionGreeks


def _log_forward_cumulants(
    params: HestonParams, maturity: float, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditional cumulants 1 and 2 of log(S_T/F_t).

    Derived from the first two powers of z in the affine Riccati ODE:
      B' = (z*z-z)/2 + (rho*xi*z-kappa)*B + xi*xi*B*B/2,
      C' = kappa*theta*B.
    This avoids finite-differencing the characteristic function at zero.
    """
    k, th, xi, rho = params.kappa, params.theta, params.xi, params.rho
    t = maturity
    e = math.exp(-k*t)
    one_minus_e = -math.expm1(-k*t)
    a = one_minus_e/k
    integrated_variance = th*t + (variances-th)*a
    c1 = -0.5*integrated_variance

    a0 = 0.5-rho*xi/(2*k)+xi*xi/(8*k*k)
    a1 = rho*xi/(2*k)-xi*xi/(4*k*k)
    a2 = xi*xi/(8*k*k)
    b2 = a0*a + a1*t*e + a2*e*one_minus_e/k
    # Integral b2(s) ds, with an expm1-safe intermediate for short maturities.
    i0 = a0*(t-a)/k
    i1 = a1*(one_minus_e-k*t*e)/(k*k)
    i2 = a2*(a - (-math.expm1(-2*k*t))/(2*k))/k
    c2 = 2.0*(variances*b2+k*th*(i0+i1+i2))
    return c1, c2


def _heston_forward_cf_and_dv(
    u: np.ndarray, variances: np.ndarray, maturity: float, params: HestonParams
) -> tuple[np.ndarray, np.ndarray]:
    """Phi_X(u), and dPhi_X/ dv, for X=log(S_T/F_t)."""
    k, th, xi, rho = params.kappa, params.theta, params.xi, params.rho
    iu = 1j*u
    z = u*u+iu
    if xi == 0:
        a = -math.expm1(-k*maturity)/k
        ivar = th*maturity + (variances-th)*a
        D = -0.5*z*a
        return np.exp(-0.5*z*ivar), D

    b = k-rho*xi*iu
    d = np.sqrt(b*b+xi*xi*z)
    bmd = -xi*xi*z/(b+d)
    g = bmd/(b+d)
    e = np.exp(-d*maturity)
    C = k*th/(xi*xi)*(bmd*maturity - 2*(np.log1p(-g*e)-np.log1p(-g)))
    D = bmd/(xi*xi) * (-np.expm1(-d*maturity))/(1-g*e)
    return np.exp(C+D*variances), D


def _integrals(
    omega: np.ndarray, a: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic COS integrals of 1 (psi) and exp(x) (chi) on [low,high]."""
    wa = omega*(low-a)
    wb = omega*(high-a)
    safe = np.where(omega==0, 1., omega)
    psi = (np.sin(wb)-np.sin(wa))/safe
    psi[:, 0] = (high-low)[:, 0]
    chi = (
        np.exp(high)*(np.cos(wb)+omega*np.sin(wb))
        - np.exp(low)*(np.cos(wa)+omega*np.sin(wa))
    )/(1.+omega**2)
    return psi, chi


def heston_cos_greeks_strikes_batch(
    params: HestonParams, market: MarketDynamics, *,
    strikes: np.ndarray, maturity: float,
    spots: np.ndarray, variances: np.ndarray,
    option_type: str = "call", n_terms: int = 512, L: float = 24.0,
    chunk_size: int = 128,
) -> OptionGreeks:
    """Batch COS prices/Greeks, sharing characteristic terms among strikes.

    Output arrays have shape (*broadcast(spots,variances).shape, len(strikes)).
    No fallback to pricing.py; numerical failures are reported, not hidden.
    The [a,b] range is based on exact c1,c2; N and L convergence are required.
    """
    if market.measure != 'Q':
        raise ValueError('COS Heston pricing requires Q dynamics')
    if option_type not in ('call','put'):
        raise ValueError('option_type must be call or put')
    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError('maturity must be finite and positive')
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 16:
        raise ValueError('n_terms must be an integer >= 16')
    if not math.isfinite(L) or L <= 0:
        raise ValueError('L must be finite and positive')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size,int) or chunk_size < 1:
        raise ValueError('chunk_size must be positive integer')
    kk = np.asarray(strikes, dtype=float)
    if kk.ndim != 1 or len(kk)==0 or not np.all(np.isfinite(kk)) or np.any(kk<=0):
        raise ValueError('strikes must be nonempty positive finite 1D array')
    ss, vv = np.broadcast_arrays(np.asarray(spots,dtype=float),np.asarray(variances,dtype=float))
    if np.any(~np.isfinite(ss)) or np.any(ss<=0) or np.any(~np.isfinite(vv)) or np.any(vv<0):
        raise ValueError('invalid spots or variances')
    original_shape=ss.shape
    sflat=ss.reshape(-1)
    vflat=vv.reshape(-1)
    n=len(sflat)
    result = {name:np.empty((n,len(kk)),dtype=float) for name in (
        'price','delta','gamma','variance_sensitivity','initial_volatility_vega')}
    r,q=market.risk_free_rate,market.dividend_yield
    discount_r=math.exp(-r*maturity)
    discount_q=math.exp(-q*maturity)
    forward_multiplier=math.exp((r-q)*maturity)
    idx=np.arange(n_terms,dtype=float)[None,:]
    for start in range(0,n,chunk_size):
        stop=min(start+chunk_size,n)
        sc=sflat[start:stop,None]
        vc=vflat[start:stop,None]
        fwd=sc*forward_multiplier
        c1,c2=_log_forward_cumulants(params,maturity,vc)
        if np.any(~np.isfinite(c2)) or np.any(c2 <= 0):
            raise FloatingPointError('COS: nonpositive or nonfinite conditional variance')
        halfwidth=L*np.sqrt(c2)
        a=c1-halfwidth
        b=c1+halfwidth
        width=2*halfwidth
        omega=idx*math.pi/width
        phi, D=_heston_forward_cf_and_dv(omega,vc,maturity,params)
        phase=phi*np.exp(-1j*omega*a)
        coeff=(2/width)*phase.real
        coeff_v=(2/width)*(phase*D).real
        coeff[:,0]*=0.5
        coeff_v[:,0]*=0.5
        if np.any(~np.isfinite(coeff)) or np.any(~np.isfinite(coeff_v)):
            raise FloatingPointError('COS: nonfinite density coefficients')

        for j,strike in enumerate(kk):
            boundary=np.log(strike/fwd)
            d=np.clip(boundary,a,b)
            # Evaluate only the OTM payoff, avoiding subtracting large numbers.
            use_put=(fwd>=strike).reshape(-1)
            low=np.where(use_put[:,None],a,d)
            high=np.where(use_put[:,None],d,b)
            psi,chi=_integrals(omega,a,low,high)
            prob=np.sum(coeff*psi,axis=1)
            tilted=np.sum(coeff*chi,axis=1)
            vprob=np.sum(coeff_v*psi,axis=1)
            vtilted=np.sum(coeff_v*chi,axis=1)
            ff=fwd[:,0]
            s=sc[:,0]
            if option_type=='call':
                price=np.where(use_put,
                    discount_r*(strike*prob-ff*tilted)+s*discount_q-strike*discount_r,
                    discount_r*(ff*tilted-strike*prob))
                delta=np.where(use_put,discount_q*(1-tilted),discount_q*tilted)
                dvar=np.where(use_put,
                    discount_r*(strike*vprob-ff*vtilted),
                    discount_r*(ff*vtilted-strike*vprob))
            else:
                price=np.where(use_put,
                    discount_r*(strike*prob-ff*tilted),
                    discount_r*(ff*tilted-strike*prob)-s*discount_q+strike*discount_r)
                delta=np.where(use_put,-discount_q*tilted,discount_q*(tilted-1))
                dvar=np.where(use_put,
                    discount_r*(strike*vprob-ff*vtilted),
                    discount_r*(ff*vtilted-strike*vprob))
            # Spot Gamma is identical for put and call. The series reconstructs
            # the *density* at x=log(K/F), holding current variance fixed.
            density=np.sum(coeff*np.cos(omega*(d-a)),axis=1)
            gamma=discount_q*np.exp(boundary[:,0])*density/s
            # If strike lies beyond the integration support, the COS density
            # has no mass there; the corresponding gamma is zero.
            gamma=np.where((boundary[:,0]<a[:,0]) | (boundary[:,0]>b[:,0]), 0., gamma)
            lower=np.maximum(s*discount_q-strike*discount_r,0.) if option_type=='call' else np.maximum(strike*discount_r-s*discount_q,0.)
            upper=s*discount_q if option_type=='call' else np.full_like(s,strike*discount_r)
            # Reject material arbitrage violations; only remove roundoff noise.
            tolerance=1e-5
            if (np.any(~np.isfinite(price)) or np.any(~np.isfinite(delta)) or
                np.any(~np.isfinite(gamma)) or np.any(~np.isfinite(dvar)) or
                np.any(price<lower-tolerance) or np.any(price>upper+tolerance) or
                np.any(gamma< -tolerance) or
                np.any(delta < (-discount_q-tolerance if option_type=='put' else -tolerance)) or
                np.any(delta > (tolerance if option_type=='put' else discount_q+tolerance))):
                bad=int(np.flatnonzero((price<lower-tolerance)|(price>upper+tolerance)|(gamma< -tolerance))[0]) if np.any((price<lower-tolerance)|(price>upper+tolerance)|(gamma< -tolerance)) else 0
                raise FloatingPointError(
                    f'COS no-arbitrage/finite violation at xi={params.xi}, S={s[bad]:.8g}, '
                    f'v={vc[bad,0]:.8g}, K={strike:.8g}, T={maturity:.8g}, '
                    f'price={price[bad]:.8g}, delta={delta[bad]:.8g}, gamma={gamma[bad]:.8g}; '
                    f'check convergence in N and L'
                )
            result['price'][start:stop,j]=np.maximum(price,lower)
            result['delta'][start:stop,j]=np.clip(delta,-discount_q if option_type=='put' else 0.,0. if option_type=='put' else discount_q)
            result['gamma'][start:stop,j]=np.maximum(gamma,0.)
            result['variance_sensitivity'][start:stop,j]=dvar
            result['initial_volatility_vega'][start:stop,j]=2*np.sqrt(vc[:,0])*dvar
    return OptionGreeks(**{name:arr.reshape(original_shape+(len(kk),)) for name,arr in result.items()})


def heston_cos_greeks_batch(
    params: HestonParams,market: MarketDynamics, *,
    strike: float,maturity: float,spots: np.ndarray,variances: np.ndarray,
    option_type: str='call',n_terms: int=512,L: float=24.0,chunk_size:int=128,
) -> OptionGreeks:
    """Single-strike convenience wrapper with reference batch output shape."""
    out=heston_cos_greeks_strikes_batch(
        params,market,strikes=np.array([strike]),maturity=maturity,
        spots=spots,variances=variances,option_type=option_type,
        n_terms=n_terms,L=L,chunk_size=chunk_size)
    return OptionGreeks(**{name:np.squeeze(getattr(out,name),axis=-1) for name in
                           ('price','delta','gamma','variance_sensitivity','initial_volatility_vega')})
