"""Before/after audit on SYNTHETIC archetype companies (NOT real tickers; live data is unreachable from the build sandbox).
Each archetype has typical large-cap magnitudes and is priced at a market-typical forward P/E, so a sound engine should
land in the neighborhood of the price, not systematically far below it.   Run: python audit/before_after.py"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import research as R, research_v1 as V1
src = open(os.path.join(os.path.dirname(HERE), "app.py")).read().rsplit("\nmain()", 1)[0]
ns = {}; exec(src, ns)
S2C, CYC = ns["S2C"], ns["CYCLICAL"]

# name, sector, industry, country, revenue $B, growth, op margin, shares B, net debt $B, beta, fwd P/E
ARCH = [("Mega-cap software", "Technology", "Software - Infrastructure", "United States", 250, .14, .45, 7.4, -40, .9, 30),
        ("AI semiconductor leader", "Technology", "Semiconductors", "United States", 120, .45, .62, 24, -30, 1.7, 30),
        ("Cyclical semiconductor", "Technology", "Semiconductors", "United States", 30, .12, .22, 1.1, 5, 1.4, 16),
        ("Hyperscaler / ad platform", "Communication Services", "Internet Content & Information", "United States", 330, .13, .32, 12.2, -70, 1.05, 22),
        ("Mid-cap growth software", "Technology", "Software - Application", "United States", 6, .22, .12, .30, -1, 1.2, 35),
        ("Payments network", "Financial Services", "Credit Services", "United States", 38, .10, .67, 1.95, 3, .95, 27),
        ("Consumer staples", "Consumer Defensive", "Household & Personal Products", "United States", 100, .04, .15, 2.5, 20, .6, 22),
        ("Consumer discretionary retail", "Consumer Cyclical", "Specialty Retail", "United States", 200, .08, .09, 1.0, 15, 1.1, 22),
        ("Regulated utility", "Utilities", "Utilities - Regulated Electric", "United States", 30, .05, .26, .78, 55, .5, 18),
        ("Power producer (AI demand)", "Utilities", "Utilities - Independent Power Producers", "United States", 20, .20, .30, .35, 15, 1.1, 22),
        ("Unprofitable growth software", "Technology", "Software - Application", "United States", 2, .35, -.05, .25, -1, 1.8, None),
        ("Diversified bank", "Financial Services", "Banks - Diversified", "United States", 0, 0, 0, 3.0, 0, 1.1, None),
        ("Large pharma", "Healthcare", "Drug Manufacturers - General", "United States", 60, .06, .30, 2.0, 40, .5, 15),
        ("Industrial cyclical", "Industrials", "Specialty Industrial Machinery", "United States", 50, .05, .15, .6, 10, 1.2, 20),
        ("Foreign ADR (TWD statements)", "Technology", "Semiconductors", "Taiwan", 90, .25, .45, 5.2, -25, 1.2, 22)]


def build(a):
    n, sec, ind, ctry, rev, g, m, sh, nd_b, beta, pe = a
    B = 1e9
    if ind.startswith("Banks"):
        bvps, roe, price = 95.0, .14, 1.7 * 95.0
        info = dict(shortName=n, sector=sec, industry=ind, country=ctry, currency="USD", financialCurrency="USD", currentPrice=price, marketCap=price * sh * B,
                    sharesOutstanding=sh * B, bookValue=bvps, returnOnEquity=roe, beta=beta, profitMargins=.25, revenueGrowth=.05, earningsGrowth=.06,
                    recommendationMean=2.3, forwardPE=11, trailingPE=12, priceToBook=1.7, totalDebt=0, totalCash=0)
        return info, None, None, price
    eps = (rev * (1 + g) * m - .05 * max(nd_b, 0)) * .8 / sh  # after interest on net debt
    price = pe * eps if pe else (8 * rev * B - nd_b * B) / (sh * B)
    debt = max(nd_b, 0) + 5; cash = debt - nd_b
    info = dict(shortName=n, sector=sec, industry=ind, country=ctry, currency="USD", financialCurrency="TWD" if ctry == "Taiwan" else "USD",
                currentPrice=price, marketCap=price * sh * B, sharesOutstanding=sh * B, enterpriseValue=price * sh * B + nd_b * B,
                totalRevenue=rev * B, operatingMargins=m, profitMargins=m * .8, grossMargins=min(m + .25, .8), revenueGrowth=g, earningsGrowth=g,
                returnOnEquity=.25 if m > 0 else -.1, bookValue=rev * .6 / sh, returnOnAssets=.10 if m > 0 else -.05, currentRatio=1.5, debtToEquity=60, recommendationMean=2.1,
                beta=beta, totalDebt=debt * B, totalCash=cash * B, forwardEps=eps if pe else None, forwardPE=pe, trailingPE=pe * 1.1 if pe else None,
                priceToSalesTrailing12Months=price * sh / rev if rev else None, freeCashflow=rev * B * max(m, 0) * .65,
                targetMeanPrice=price * 1.1, targetHighPrice=price * 1.4, targetLowPrice=price * .8)
    ks = range(4); cols = pd.to_datetime(["2025-09-30", "2024-09-30", "2023-09-30", "2022-09-30"])
    rv = [rev * B / (1 + g) ** k for k in ks]; mm = [m * (1 - .04 * k) if g > .1 else m * (1 + .10 * k) if g < 0 else m for k in ks]  # growers expanding; decliners deteriorating
    inc = pd.DataFrame({c: {"Total Revenue": r, "Operating Income": r * x, "Net Income": r * x * .8, "Gross Profit": r * (m + .25)} for c, r, x in zip(cols, rv, mm)}).astype(float)
    cf = pd.DataFrame({c: {"Free Cash Flow": r * x * .7, "Operating Cash Flow": r * x * .85, "Stock Based Compensation": r * .03} for c, r, x in zip(cols, rv, mm)}).astype(float)
    bs = pd.DataFrame({c: {"Total Debt": debt * B, "Capital Lease Obligations": 4 * B, "Cash Cash Equivalents And Short Term Investments": cash * B,
                           "Total Assets": rev * B * 1.5, "Stockholders Equity": rev * B * .6, "Ordinary Shares Number": sh * B} for c in cols}).astype(float)
    return info, inc, cf, bs, price


def scores_for(info, inc, cf, bs, price):
    r = np.random.RandomState(abs(hash(info["shortName"])) % 1000)
    idx = pd.bdate_range(end="2026-10-09", periods=2600)
    c = pd.Series(price * np.exp(np.cumsum(r.normal(.0003, .014, len(idx)))), index=idx); c = c * price / c.iloc[-1]
    prof = ns["profile_for"](info, info["sector"])
    df = ns["build_metrics"](info, pd.DataFrame({"Close": c}), inc, cf, bs, prof)
    cs = df.groupby("Category")["Score"].mean().to_dict()
    q = float(np.clip((cs.get("Profitability", 0) + cs.get("Growth", 0) + cs.get("Financial health", 0)) / 3, -1, 1))
    return df, cs, q


def old_engine(info, inc, price, cs, q):
    """Faithful replica of the v1 pipeline (research_v1 + the old inline defaults in app_v1.research_ui)."""
    sec = info["sector"]
    if sec in ("Financial Services", "Real Estate"):
        ke = .042 + max(.6, min(2.0, info["beta"])) * .045
        if info.get("bookValue") is None: return None
        sc_ = V1.fin_scenarios(info["bookValue"], info["returnOnEquity"], ke, .025, q)
    else:
        rev = R.getrow(inc, ["Total Revenue"]); oi = R.getrow(inc, ["Operating Income"])
        m0 = info["operatingMargins"]; avg = float((oi / rev).dropna().mean())
        ke = .042 + float(np.clip(info["beta"], .6, 2.0)) * .045
        wd = info["totalDebt"] / (info["totalDebt"] + info["marketCap"])
        wacc = float(np.clip((1 - wd) * ke + wd * .062 * .79, .065, .14))
        p = dict(rev0=info["totalRevenue"], g1=float(np.clip(info["revenueGrowth"], -.05, .35)), m0=m0, mt=float(np.clip(.6 * m0 + .4 * avg, -.1, .45)),
                 wacc=wacc, tg=.025, tax=.21, s2c=S2C.get(sec, 1.5), net_debt=info["totalDebt"] - info["totalCash"], shares=info["sharesOutstanding"])
        if info["financialCurrency"] != info["currency"]: pass  # v1 had no currency guard: it silently mixed TWD and USD
        sc_ = V1.scenarios(p, q)
    pw = sum(v["prob"] * v["fv"] for v in sc_.values()); ret = V1.expected_returns(price, pw, ke)
    val = dict(price=price, upside=pw / price - 1, exp5=ret[5], ke=ke, bear=sc_["Bear"]["fv"], base=sc_["Base"]["fv"], bull=sc_["Bull"]["fv"], plaus="Reasonable")
    s = V1.make_scores(cs, val, 45, 0); tot, _ = V1.overall(s, V1.DEFAULT_W)
    return pw, V1.verdict(tot, ret[5], ke)[0], sc_["Base"]["fv"]


def new_engine(info, inc, bs, price, cs, q):
    sec = info["sector"]; kind_pb = R.use_book_value(info, sec)
    d = R.derive(info, price, inc, bs, sec, .0425, info["revenueGrowth"] if not kind_pb else None, sec in CYC, S2C,
                 sum(ns["profile_for"](info, sec)["over"].get("Operating margin", (.12, .12))) / 2)
    inp = dict(price=price, q=q, ke=d["ke"], hist_g=d["hist_g"], cyc=sec in CYC, dcf_w=(.2 if sec == "Utilities" else .4 if sec in CYC else .6), fwd_eps=info.get("forwardEps") if d["cur_ok"] else None, pe_rng=ns["profile_for"](info, sec)["exit"])
    if kind_pb: inp["pb"] = dict(bvps=info["bookValue"], roe=info["returnOnEquity"], g=.025)
    elif d["dcf_ok"]: inp["p"] = dict(rev0=d["rev0"], g1=d["g1"], m0=d["m0"], mt=d["mt"], wacc=d["wacc"], tg=d["tg"], tax=.21, s2c=d["s2c"], net_debt=d["nd"] + d["mi"], shares=d["shares"])
    Vv = R.valuate(inp)
    if not Vv["scen"]: return None, d
    s = R.make_scores(cs, Vv["val"], 45, 2 if d["speculative"] else 0, Vv["disagree"]); tot, _ = R.overall(s, R.DEFAULT_W)
    return (Vv["pw"], R.verdict(tot, Vv["ret"][5], d["ke"], s["Forecast confidence"], Vv["disagree"])[0],
            Vv["base_dcf"]["per_share"] if Vv["base_dcf"] else Vv["scen"]["Base"]["fv"], Vv["mult"]["Base"]["fv"] if Vv["mult"] else None), d


rows = []
for a in ARCH:
    built = build(a); info, inc, cf, bs, price = (built if len(built) == 5 else (built[0], None, None, None, built[3]))
    if inc is None:
        df = pd.DataFrame([{"Category": "Profitability", "Score": .3}, {"Category": "Financial health", "Score": .2}, {"Category": "Growth", "Score": 0.0},
                           {"Category": "Valuation", "Score": 0.0}, {"Category": "Risk", "Score": 0.0}])
        cs = df.groupby("Category")["Score"].mean().to_dict(); q = .17
    else:
        df, cs, q = scores_for(info, inc, cf, bs, price)
    old = old_engine(info, inc, price, cs, q); new, d = new_engine(info, inc, bs, price, cs, q)
    f = lambda x: "n/a" if x is None else f"${x:,.0f}" if x >= 100 else f"${x:,.1f}"
    rows.append({"Archetype": a[0], "Price": f(price), "OLD fair value": f(old[0]) if old else "n/a", "OLD / price": f"{old[0] / price:.2f}x" if old else "n/a",
                 "OLD verdict": old[1] if old else "n/a",
                 "NEW fair value": f(new[0]) if new else ("DCF disabled" if d["issues"] else "n/a"), "NEW / price": f"{new[0] / price:.2f}x" if new else "n/a",
                 "NEW verdict": new[1] if new else "not rated (data issue)", "NEW DCF-only": f(new[2]) if new else "n/a", "NEW multiples": f(new[3]) if new and len(new) > 3 else "n/a"})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
ov = [float(r["OLD / price"][:-1]) for r in rows if r["OLD / price"] != "n/a"]; nv = [float(r["NEW / price"][:-1]) for r in rows if r["NEW / price"] != "n/a"]
print(f"\nMedian fair value / price:  OLD {np.median(ov):.2f}x   NEW {np.median(nv):.2f}x")
for tag in ("OLD verdict", "NEW verdict"): print(tag, t[tag].value_counts().to_dict())
with open(os.path.join(HERE, "before_after.md"), "w") as fh:
    fh.write("# Before / after on synthetic archetypes (not real tickers)\n\n" + t.to_markdown(index=False) + f"\n\nMedian fair value / price: OLD {np.median(ov):.2f}x, NEW {np.median(nv):.2f}x\n")
