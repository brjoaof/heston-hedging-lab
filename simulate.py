"""Generate Heston trajectories, option books, initial values and expiry cashflows."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import sample_books, expiry_cashflows
from pricing import price_book
from volatility import build_volatility_lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("results/simulations"))
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--n-books", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sim = config["simulation"]
    book_cfg = config["books"]
    if args.n_paths is not None:
        sim["n_paths"] = args.n_paths
    if args.n_books is not None:
        book_cfg["n_books"] = args.n_books

    params = HestonParams(**config["heston"])
    market = MarketDynamics(
        measure=sim["measure"],
        spot0=sim["spot0"],
        risk_free_rate=sim["risk_free_rate"],
        dividend_yield=sim["dividend_yield"],
        physical_total_return=sim.get("physical_total_return"),
    )
    paths = simulate_heston(
        params, market,
        n_paths=sim["n_paths"],
        n_steps=sim["n_steps"],
        trading_days_per_year=sim["trading_days_per_year"],
        seed=sim["seed"],
    )
    portfolios = sample_books(
        spot_reference=market.spot0,
        n_books=book_cfg["n_books"],
        sizes=tuple(book_cfg["sizes"]),
        expiries_days=tuple(book_cfg["expiries_days"]),
        strike_moneyness_range=tuple(book_cfg["strike_moneyness_range"]),
        max_abs_quantity=book_cfg["max_abs_quantity"],
        seed=book_cfg["seed"],
    )
    if any(contract.maturity_days > sim["n_steps"]
           for book in portfolios for contract in book.contracts):
        raise ValueError("A contract maturity exceeds the simulation horizon")

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "paths.npz",
        time_years=paths.time_years,
        spot=paths.spot,
        variance=paths.variance,
        raw_variance=paths.raw_variance,
    )
    with (args.output / "portfolios.jsonl").open("w", encoding="utf-8") as file:
        for book in portfolios:
            file.write(json.dumps({
                "book_id": book.book_id,
                "contracts": [asdict(contract) for contract in book.contracts],
            }) + "\n")

    # Signed contractual payoffs only; not option prices or hedging P&L.
    cashflows = np.stack([
        expiry_cashflows(book, paths.spot) for book in portfolios
    ])
    np.savez_compressed(args.output / "expiry_cashflows.npz", cashflows=cashflows)

    # Initial Q model values and state sensitivities. A P-path scenario alone
    # is insufficient for pricing: it requires separately specified Q inputs.
    prices_file = args.output / "initial_prices.jsonl"
    if market.measure == "Q":
        with prices_file.open("w", encoding="utf-8") as file:
            for book in portfolios:
                greeks = price_book(
                    book, params, market,
                    trading_days_per_year=sim["trading_days_per_year"],
                )
                file.write(json.dumps({"book_id": book.book_id, **asdict(greeks)}) + "\n")
    elif prices_file.exists():
        prices_file.unlink()  # prevent mixing stale Q prices with P simulations

    # For fixed Q parameters and log(K/F), the surface is a function of
    # current variance only. Precompute that 1D dependence once, then
    # evaluate all path states without repeated Fourier integrations.
    surface_file = args.output / "implied_vol_surface.npz"
    path_surface_file = args.output / "implied_vol_paths.npz"
    surface_config = config.get("volatility_surface")
    if market.measure == "Q" and surface_config is not None:
        lookup = build_volatility_lookup(
            params, market,
            maturities_days=surface_config["maturities_days"],
            log_moneyness=surface_config["log_moneyness"],
            variance_nodes=surface_config["variance_nodes"],
            trading_days_per_year=sim["trading_days_per_year"],
        )
        iv_paths = lookup.evaluate(paths.variance)
        # Keep the initial surface filename and data layout unchanged.
        initial_iv = lookup.evaluate(np.asarray(params.v0))
        tenors = lookup.maturities_days / sim["trading_days_per_year"]
        strikes = market.spot0 * np.exp(
            (market.risk_free_rate - market.dividend_yield) * tenors[:, None]
            + lookup.log_moneyness[None, :]
        )
        np.savez_compressed(
            surface_file,
            maturities_days=lookup.maturities_days,
            log_moneyness=lookup.log_moneyness,
            strikes=strikes,
            implied_volatility=initial_iv,
            spot=market.spot0,
            variance=params.v0,
        )
        # Dynamic strikes for every path and day are reconstructed from
        # paths.spot, tenors and log_moneyness (not stored redundantly).
        np.savez_compressed(
            path_surface_file,
            implied_volatility=iv_paths,
            maturities_days=lookup.maturities_days,
            log_moneyness=lookup.log_moneyness,
            variance_nodes=lookup.variance_nodes,
            lookup_implied_volatility=lookup.implied_volatility,
        )
    else:
        for obsolete in (surface_file, path_surface_file):
            if obsolete.exists():
                obsolete.unlink()  # never leave stale Q results from a previous run

    horizon = float(paths.time_years[-1])
    adjusted_spot = paths.spot[:, -1] * np.exp(-market.spot_drift * horizon)
    summary = {
        "config": config,
        "measure": paths.measure,
        "path_shape": list(paths.spot.shape),
        "number_of_portfolios": len(portfolios),
        "cashflow_shape": list(cashflows.shape),
        "initial_prices_computed_under_Q": market.measure == "Q",
        "dynamic_iv_computed_under_Q": market.measure == "Q" and surface_config is not None,
        "dynamic_iv_shape": list(iv_paths.shape) if market.measure == "Q" and surface_config is not None else None,
        "feller_ratio": params.feller_ratio,
        "fraction_negative_raw_variance": float(np.mean(paths.raw_variance < 0)),
        "mean_terminal_variance": float(paths.raw_variance[:, -1].mean()),
        "expected_terminal_variance": params.expected_variance(horizon),
        "mean_drift_adjusted_spot": float(adjusted_spot.mean()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Generated {len(portfolios)} portfolios and {len(paths.spot)} price paths")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
