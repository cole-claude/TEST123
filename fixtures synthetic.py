"""SYNTHETIC test companies (typical magnitudes, deterministic). Used only by tests; never imported by the app and never shown to users."""
import zlib
import numpy as np
import pandas as pd

B = 1e9
A = [  # name, sector, industry, rev $B, growth, gross margin, op margin, shares B, cash $B, debt $B, beta, fwd P/E or EV/Sales, capex % rev, margin slope/yr, dilution/yr
    dict(name="Mature consumer staples", sector="Consumer Defensive", industry="Household & Personal Products", rev=100, g=.04, gm=.35, m=.15, sh=2.5, cash=8, debt=28, beta=.6, pe=22, capex=.04, slope=0, dil=-.01),
    dict(name="Mature software compounder", sector="Technology", industry="Software - Infrastructure", rev=250, g=.14, gm=.70, m=.45, sh=7.4, cash=80, debt=40, beta=.9, pe=30, capex=.12, slope=.01, dil=-.005),
    dict(name="High-growth profitable platform, rich multiple", sector="Technology", industry="Software - Application", rev=10, g=.25, gm=.50, m=.13, sh=1.3, cash=6, debt=1, beta=2.6, pe=75, capex=.02, slope=.03, dil=.01),
    dict(name="AI semiconductor leader", sector="Technology", industry="Semiconductors", rev=120, g=.45, gm=.75, m=.62, sh=24, cash=40, debt=10, beta=1.7, pe=30, capex=.05, slope=.05, dil=0),
    dict(name="Cyclical semiconductor", sector="Technology", industry="Semiconductors", rev=30, g=.12, gm=.50, m=.22, sh=1.1, cash=3, debt=8, beta=1.4, pe=16, capex=.15, slope=.04, dil=0),
    dict(name="Payments network", sector="Financial Services", industry="Credit Services", rev=38, g=.10, gm=.80, m=.67, sh=1.95, cash=17, debt=20, beta=.95, pe=27, capex=.04, slope=0, dil=-.02),
    dict(name="Regional bank", sector="Financial Services", industry="Banks - Regional", bank=True, sh=.5, beta=1.0),
    dict(name="Regulated utility", sector="Utilities", industry="Utilities - Regulated Electric", rev=30, g=.05, gm=.40, m=.26, sh=.78, cash=1, debt=56, beta=.5, pe=18, capex=.45, slope=0, dil=.02),
    dict(name="Oil and gas producer", sector="Energy", industry="Oil & Gas E&P", rev=40, g=.02, gm=.45, m=.25, sh=1.2, cash=5, debt=10, beta=1.3, pe=11, capex=.30, slope=-.03, dil=-.02),
    dict(name="AI infrastructure, loss-making, funded", sector="Technology", industry="Information Technology Services", rev=1.2, g=2.0, gm=.45, m=-.8, sh=.24, cash=5.5, debt=.8, beta=2.2, evs=9, capex=1.8, slope=.6, dil=.10),
    dict(name="AI infrastructure, loss-making, short runway", sector="Technology", industry="Information Technology Services", rev=1.2, g=2.0, gm=.45, m=-.8, sh=.24, cash=2.0, debt=.8, beta=2.2, evs=9, capex=1.8, slope=.6, dil=.10),
    dict(name="Emerging software, loss-making, well funded", sector="Technology", industry="Software - Application", rev=2, g=.35, gm=.70, m=-.05, sh=.25, cash=1, debt=0, beta=1.8, evs=8, capex=.03, slope=.12, dil=.03),
    dict(name="Speculative pre-scale technology", sector="Technology", industry="Software - Application", rev=.05, g=.8, gm=.20, m=-3.0, sh=.1, cash=.15, debt=.05, beta=2.5, evs=20, capex=.5, slope=1.0, dil=.15),
    dict(name="Turnaround, loss-making", sector="Industrials", industry="Specialty Industrial Machinery", rev=20, g=-.04, gm=.22, m=-.02, sh=.5, cash=1, debt=6, beta=1.3, evs=.6, capex=.04, slope=-.04, dil=.01),
    dict(name="Excellent company, excessive valuation", sector="Technology", industry="Software - Infrastructure", rev=80, g=.20, gm=.75, m=.40, sh=4, cash=10, debt=0, beta=1.1, pe=60, capex=.04, slope=.02, dil=0),
    dict(name="Cheap, deteriorating, still profitable", sector="Industrials", industry="Specialty Industrial Machinery", rev=20, g=-.04, gm=.25, m=.06, sh=.5, cash=1, debt=8, beta=1.3, pe=8, capex=.04, slope=-.03, dil=0, weak=True),
]


def _hist(name, price, drift=.0004, weak=False):
    r = np.random.RandomState(zlib.crc32(name.encode()) % 100000)
    n = 2700
    d = np.full(n, drift); d[-126:] = -.0025 if weak else drift
    c = pd.Series(np.exp(np.cumsum(r.normal(d, .014))), index=pd.bdate_range(end="2026-10-09", periods=n))
    return pd.DataFrame({"Close": c * price / c.iloc[-1]})


def build(a):
    """Returns dict(info, hist, inc, cf, bs, price, est). Numbers are internally consistent, not real."""
    if a.get("bank"):
        bv, roe = 40.0, .12; price = 1.2 * bv
        info = dict(shortName=a["name"], sector=a["sector"], industry=a["industry"], country="United States", currency="USD", financialCurrency="USD", currentPrice=price,
                    marketCap=price * a["sh"] * B, sharesOutstanding=a["sh"] * B, bookValue=bv, returnOnEquity=roe, payoutRatio=.35, beta=a["beta"], profitMargins=.25,
                    revenueGrowth=.05, earningsGrowth=.06, recommendationMean=2.4, priceToBook=1.2, forwardPE=10, trailingPE=10.5, totalDebt=0, totalCash=0, forwardEps=price / 10, trailingEps=price / 10.5)
        return dict(info=info, hist=_hist(a["name"], price), inc=None, cf=None, bs=None, price=price, est={})
    rev, g, gm, m, sh, cash, debt = a["rev"] * B, a["g"], a["gm"], a["m"], a["sh"] * B, a["cash"] * B, a["debt"] * B
    intr = .05 * max(debt - cash, 0)
    ni_f = (rev * (1 + g) * m - intr) * (.8 if m > 0 else 1)
    eps_f = ni_f / sh
    price = a["pe"] * eps_f if a.get("pe") else (a["evs"] * rev - debt + cash) / sh
    mc = price * sh
    fcf0 = rev * m * (.7 if m > 0 else 1.0) - a["capex"] * rev
    ni_t = (rev * m - intr) * (.8 if m > 0 else 1)
    ebitda = rev * (m + .05)
    info = dict(shortName=a["name"], sector=a["sector"], industry=a["industry"], country="United States", currency="USD", financialCurrency="USD", currentPrice=price,
                marketCap=mc, sharesOutstanding=sh, enterpriseValue=mc + debt - cash, totalRevenue=rev, operatingMargins=m, profitMargins=ni_t / rev, grossMargins=gm,
                revenueGrowth=g, earningsGrowth=g if m > 0 else None, forwardEps=eps_f if eps_f > 0 else None, trailingEps=ni_t / sh, forwardPE=price / eps_f if eps_f > 0 else None,
                trailingPE=price / (ni_t / sh) if ni_t > 0 else None, totalDebt=debt, totalCash=cash, freeCashflow=fcf0, ebitda=ebitda, enterpriseToEbitda=(mc + debt - cash) / ebitda if ebitda > 0 else None,
                enterpriseToRevenue=(mc + debt - cash) / rev, priceToSalesTrailing12Months=mc / rev, beta=a["beta"], recommendationMean=2.2, returnOnEquity=.2 if m > 0 else -.3,
                returnOnAssets=.08 if m > 0 else -.15, currentRatio=1.6, debtToEquity=60, bookValue=max(rev * .5, cash) / sh, longBusinessSummary="Synthetic test company.")
    cols = [f"{y}-12-31" for y in (2025, 2024, 2023, 2022)]
    I, C, S = {}, {}, {}
    for k, c in enumerate(cols):
        rv, mk, gk = rev / (1 + g) ** k, m - a["slope"] * k, gm - .02 * k * (g > .1)
        ebit = rv * mk
        fcf = ebit * (.7 if ebit > 0 else 1.0) - a["capex"] * rv
        I[c] = {"Total Revenue": rv, "Gross Profit": rv * gk, "Operating Income": ebit, "Net Income": ebit * (.8 if ebit > 0 else 1.0), "EBIT": ebit, "Interest Expense": -intr}
        C[c] = {"Operating Cash Flow": fcf + a["capex"] * rv, "Free Cash Flow": fcf, "Capital Expenditure": -a["capex"] * rv, "Stock Based Compensation": .03 * rv}
        S[c] = {"Total Debt": debt, "Cash Cash Equivalents And Short Term Investments": cash, "Total Assets": rv * 1.5 + cash, "Stockholders Equity": max(rv * .5, cash),
                "Goodwill": rv * .05, "Ordinary Shares Number": sh / (1 + a["dil"]) ** k}
    return dict(info=info, hist=_hist(a["name"], price, weak=a.get("weak", False)), inc=pd.DataFrame(I).astype(float), cf=pd.DataFrame(C).astype(float),
                bs=pd.DataFrame(S).astype(float), price=price, est={})
