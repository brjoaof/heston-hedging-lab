"""Evaluate simple hedging strategies on P-Heston paths using Q option prices."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import sample_books
from hedging import value_book_paths, simulate_hedge, summarize_hedge
from volatility import build_volatility_lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("results/hedging"))
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--n-books", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sim, book_cfg, hedge = config["simulation"], config["books"], config["hedging"]
    q_params = HestonParams(**config["heston"])
    p_params = HestonParams(**hedge["physical_heston"])
    q_market = MarketDynamics(
        "Q", sim["spot0"], sim["risk_free_rate"], sim["dividend_yield"]
    )
    p_market = MarketDynamics(
        "P", sim["spot0"], sim["risk_free_rate"], sim["dividend_yield"],
        physical_total_return=hedge["physical_total_return"],
    )
    paths = simulate_heston(
        p_params, p_market, n_paths=args.n_paths or hedge["n_paths"],
        n_steps=sim["n_steps"], trading_days_per_year=sim["trading_days_per_year"],
        seed=hedge["seed"],
    )
    books = sample_books(
        spot_reference=p_market.spot0,
        n_books=args.n_books or hedge["n_books"],
        sizes=tuple(book_cfg["sizes"]),
        expiries_days=tuple(book_cfg["expiries_days"]),
        strike_moneyness_range=tuple(book_cfg["strike_moneyness_range"]),
        max_abs_quantity=book_cfg["max_abs_quantity"], seed=book_cfg["seed"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    surface = config["volatility_surface"]
    lookup = build_volatility_lookup(
        q_params, q_market,
        maturities_days=surface["maturities_days"],
        log_moneyness=surface["log_moneyness"],
        variance_nodes=surface["variance_nodes"],
        trading_days_per_year=sim["trading_days_per_year"],
    )
    np.savez_compressed(args.output/"market_paths.npz",
        spot=paths.spot, variance=paths.variance, time_years=paths.time_years,
        implied_volatility=lookup.evaluate(paths.variance),
        maturities_days=lookup.maturities_days,
        log_moneyness=lookup.log_moneyness,
    )
    with (args.output/"portfolios.jsonl").open("w", encoding="utf-8") as f:
        for book in books:
            f.write(json.dumps({"book_id":book.book_id,
                "contracts":[asdict(c) for c in book.contracts]})+"\n")
    rows = []
    book_values = []
    metrics_by_path = []
    for book in books:
        valuation = value_book_paths(
            book, paths, q_params, q_market,
            trading_days_per_year=sim["trading_days_per_year"],
        )
        book_values.append(valuation)
        strategy_results = []
        for strategy in ("no_hedge", "delta_daily", "delta_periodic", "delta_band"):
            result = simulate_hedge(
                paths, valuation, p_market, strategy=strategy,
                proportional_cost=hedge["proportional_cost"],
                fixed_cost=hedge["fixed_cost"],
                band_width=hedge["band_width"],
                min_trade=hedge["min_trade"],
                rebalance_every=hedge["rebalance_every"],
                trading_days_per_year=sim["trading_days_per_year"],
            )
            strategy_results.append(result)
            rows.append({"book_id":book.book_id, "n_contracts":book.n_contracts,
                         "strategy":strategy, **summarize_hedge(result)})
        metrics_by_path.append(strategy_results)
    np.savez_compressed(args.output/"book_paths.npz",
        value=np.stack([b.value for b in book_values]),
        delta=np.stack([b.delta for b in book_values]),
        gamma=np.stack([b.gamma for b in book_values]),
        variance_sensitivity=np.stack([b.variance_sensitivity for b in book_values]),
        settlements=np.stack([b.settlements for b in book_values]),
    )
    # Axes: (book, strategy, simulated P path)
    np.savez_compressed(args.output/"hedging_results.npz",
        strategies=np.array(["no_hedge", "delta_daily", "delta_periodic", "delta_band"]),
        terminal_pnl=np.array([[r.terminal_pnl for r in rr] for rr in metrics_by_path]),
        transaction_cost=np.array([[r.transaction_cost for r in rr] for rr in metrics_by_path]),
        turnover=np.array([[r.turnover for r in rr] for rr in metrics_by_path]),
        trades=np.array([[r.trades for r in rr] for rr in metrics_by_path]),
    )
    (args.output/"metrics.json").write_text(json.dumps({
        "physical_model": asdict(p_params), "pricing_model": asdict(q_params),
        "market_P": asdict(p_market), "market_Q": asdict(q_market),
        "n_paths":len(paths.spot), "seed":hedge["seed"], "results":rows,
    }, indent=2), encoding="utf-8")
    print(f"Evaluated {len(books)} books over {len(paths.spot)} P paths")
    print(f"Metrics: {(args.output/'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
