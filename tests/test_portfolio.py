import numpy as np
import pytest
from portfolio import EuropeanOption, OptionBook, sample_books, expiry_cashflows


def test_variable_size_and_deterministic():
    a = sample_books(spot_reference=100, n_books=100, seed=31)
    b = sample_books(spot_reference=100, n_books=100, seed=31)
    assert a == b
    assert set(book.n_contracts for book in a) == {1, 3, 5, 9}
    assert all(.85 * 100 <= c.strike <= 1.15 * 100 for book in a for c in book.contracts)


def test_intermediate_expiry_cashflow():
    # One 2-unit long call at day 1 and one 1-unit short put at day 2.
    book = OptionBook("book", (
        EuropeanOption("long-call", 2, 100, 1, "call"),
        EuropeanOption("short-put", -1, 102, 2, "put"),
    ))
    spot = np.array([[100., 110., 95.], [100., 90., 105.]])
    out = expiry_cashflows(book, spot)
    np.testing.assert_array_equal(out, [[0., 20., -7.], [0., 0., 0.]])


def test_book_duplicates_rejected():
    with pytest.raises(ValueError, match="unique"):
        OptionBook("b", (
            EuropeanOption("a", 1, 100, 21),
            EuropeanOption("a", -1, 110, 42),
        ))


def test_maturity_after_horizon_rejected():
    b = OptionBook("b", (EuropeanOption("a", 1, 100, 2),))
    with pytest.raises(ValueError, match="expires beyond"):
        expiry_cashflows(b, np.array([[100., 110.]]))
