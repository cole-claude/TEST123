"""Regression tests: one per root cause found in the valuation audit. Run: python -m pytest -q"""
import math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit"))
import research as R
import research_v1 as V1  # frozen snapshot of the engine BEFORE the fixes

P = dict(rev0=1000, m0=.25, mt=.25, wacc=.09, tg=.025, tax=.21, s2c=2.0, net_debt=0, shares=100)
S2C = {"Technology": 3}
INFO = dict(marketCap=1e11, totalRevenue=2e10, operatingMargins=.25, beta=1.5, totalDebt=1e10, totalCash=5e9, currency="USD",
            financialCurrency="USD", sector="Technology", industry="Software", country="United States")


def test_bug1_growth_was_faded_from_year_one():
    assert R.dcf(g1=.15, **P)["per_share"] > V1.dcf(g1=.15, **P)["per_share"]          # growth now held for 3 years
    assert math.isclose(R.dcf(g1=.0, **{**P, "tg": 0})["per_share"], V1.dcf(g1=.0, **{**P, "tg": 0})["per_share"])  # no-growth unchanged


def test_bug2_scenarios_symmetric_and_no_wacc_double_count():
    s = R.scenarios({**P, "g1": .10}, 0)
    assert s["Bull"]["p"]["wacc"] == s["Base"]["p"]["wacc"] == s["Bear"]["p"]["wacc"]
    b, m, r = (s[k]["p"] for k in ("Bull", "Base", "Bear"))
    assert math.isclose(b["g1"] - m["g1"], m["g1"] - r["g1"]) and math.isclose(b["mt"] - m["mt"], m["mt"] - r["mt"])
    old = V1.scenarios({**P, "g1": .10}, 0)
    assert old["Bear"]["p"]["wacc"] > old["Base"]["p"]["wacc"]  # documents the old double count


def test_bug3_beta_shrinkage():
    assert math.isclose(R.shrink_beta(2.0), 1.67) and R.shrink_beta(None) == 1.0 and R.shrink_beta(.5) > .5


def test_bug4_us_operating_leases_and_minority_interest():
    cols = pd.to_datetime(["2025-12-31"])
    bs = pd.DataFrame({"Total Debt": [100.0], "Capital Lease Obligations": [30.0], "Cash Cash Equivalents And Short Term Investments": [20.0],
                       "Minority Interest": [5.0]}, index=cols).T
    nd, mi, _ = R.net_debt({"country": "United States"}, bs)
    assert nd == 50.0 and mi == 5.0                       # 100 - 30 leases - 20 cash
    assert R.net_debt({"country": "Germany"}, bs)[0] == 80.0  # IFRS filers: leases stay in debt


def test_bug5_payments_networks_not_valued_on_book_value():
    assert not R.use_book_value({"industry": "Credit Services", "totalDebt": 2e10, "marketCap": 6e11}, "Financial Services")
    assert R.use_book_value({"industry": "Credit Services", "totalDebt": 9e10, "marketCap": 6e10}, "Financial Services")  # lender
    assert R.use_book_value({"industry": "Banks - Diversified"}, "Financial Services")
    assert not R.use_book_value({"industry": "Financial Data & Stock Exchanges"}, "Financial Services")


def test_bug6_currency_mismatch_disables_dcf_and_shares_use_marketcap():
    d = R.derive({**INFO, "financialCurrency": "TWD"}, 100.0, None, None, "Technology", .0425, .1, False, S2C)
    assert not d["dcf_ok"] and not d["cur_ok"] and d["issues"]
    d = R.derive(INFO, 100.0, None, None, "Technology", .0425, .1, False, S2C)
    assert d["dcf_ok"] and math.isclose(d["shares"], 1e11 / 100.0)


def test_bug7_loss_making_growth_not_valued_at_zero():
    info = {**INFO, "operatingMargins": -.05, "totalRevenue": 2e9, "marketCap": 1.7e10, "totalDebt": 0, "totalCash": 1e9}
    d = R.derive(info, 68.0, None, None, "Technology", .0425, .35, False, S2C, .165)
    assert d["speculative"] and math.isclose(d["mt"], .165)
    p = dict(rev0=2e9, g1=d["g1"], m0=-.05, mt=d["mt"], wacc=d["wacc"], tg=d["tg"], tax=.21, s2c=3.0, net_debt=-1e9, shares=2.5e8)
    assert R.dcf(**p)["per_share"] > 10
    assert V1.dcf(**{**p, "mt": -.05})["per_share"] < 1       # old behavior: negative margin held forever => ~0


def test_bug8_risk_free_scale_guard_and_fallback():
    assert math.isclose(R.derive(INFO, 100., None, None, "Technology", .425, .1, False, S2C)["rf"], .0425)   # ^TNX quoted x10
    assert math.isclose(R.derive(INFO, 100., None, None, "Technology", .0425, .1, False, S2C)["rf"], .0425)
    d = R.derive(INFO, 100., None, None, "Technology", None, .1, False, S2C)
    assert d["rf"] == .042 and not d["rf_live"]               # labeled fallback, never silent


def test_bug9_scores_no_longer_saturate_on_modest_gaps():
    val = dict(price=100, upside=-.15, exp5=.045, ke=.095, bear=70, base=90, bull=110, plaus="Reasonable")
    new, old = R.make_scores({}, val, 45, 0), V1.make_scores({}, val, 45, 0)
    assert new["Expected return"] > old["Expected return"] > -1 and new["Expected return"] > -.5


def test_bug10_strong_ratings_require_agreement_and_confidence():
    assert R.verdict(-.6, .0, .09)[0] == "Strong Avoid"
    assert R.verdict(-.6, .0, .09, conf=.3)[0] == "Avoid"
    assert R.verdict(-.6, .0, .09, disagree=.8)[0] == "Avoid"
    assert R.verdict(.6, .2, .09, disagree=.8)[0] == "Buy"


def test_blend_widens_range_when_methods_disagree():
    d = {k: {"fv": v, "prob": p} for k, v, p in (("Bull", 120, .25), ("Base", 100, .5), ("Bear", 80, .25))}
    m = {k: {"fv": v, "prob": p} for k, v, p in (("Bull", 330, .25), ("Base", 300, .5), ("Bear", 250, .25))}
    out, gap = R.blend(d, m, .6, 0)
    assert gap > .5 and out["Bear"]["fv"] <= 80 and out["Bull"]["fv"] >= 330
    assert math.isclose(sum(v["prob"] for v in out.values()), 1.0)


def test_reconcile_flags_inconsistent_inputs():
    ck = R.reconcile({"marketCap": 100.0}, 10.0, 20.0, 0.0, 0.0, None, None)  # 10 * 20 = 200 != 100
    assert ck and not ck[0][1]


def test_missing_data_never_fabricates_a_valuation():
    d = R.derive({}, 100.0, None, None, "Technology", None, None, False, S2C)
    assert not d["dcf_ok"]
    out = R.valuate(dict(price=100.0, q=0, ke=.09, pe_rng=(18, 30), fwd_eps=None))
    assert out["scen"] is None and out["val"] is None
