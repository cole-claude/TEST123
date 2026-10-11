"""Rating-consistency matrix on 21 SYNTHETIC archetypes (not real tickers: live data is unreachable from the build sandbox).
Compares the pre-fix logic (audit/app_v2.py + research_v2.py: header ratings from the metric model, overall from R.verdict)
with the fixed single-source logic (research.recommend).   Run: python audit/rating_matrix.py"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import research as R, research_v2 as R2
g = {"__file__": os.path.join(HERE, "before_after.py")}
exec(open(os.path.join(HERE, "before_after.py")).read().split("\nrows = []")[0], g)
ARCH, build, ns = list(g["ARCH"]), g["build"], g["ns"]; S2C, CYC = ns["S2C"], ns["CYCLICAL"]
ns2 = {}; exec(open(os.path.join(HERE, "app_v2.py")).read().rsplit("\nmain()", 1)[0], ns2)   # pre-fix app logic
ARCH += [("E-commerce platform, high growth, rich multiple", "Technology", "Software - Application", "United States", 10, .25, .13, 1.3, -5, 2.6, 75),
         ("Streaming subscription", "Communication Services", "Entertainment", "United States", 45, .13, .27, .43, 10, 1.3, 32),
         ("Apparel brand, slowing growth", "Consumer Cyclical", "Footwear & Accessories", "United States", 48, .01, .12, 1.5, 4, 1.1, 24),
         ("Excellent company, excessive valuation", "Technology", "Software - Infrastructure", "United States", 80, .20, .40, 4, -10, 1.1, 60),
         ("Cheap, deteriorating fundamentals", "Industrials", "Specialty Industrial Machinery", "United States", 20, -.04, .06, .5, 8, 1.3, 8),
         ("Strong long-term, weak short-term catalysts", "Technology", "Semiconductor Equipment", "United States", 40, .18, .30, 1.0, -5, 1.3, 24)]
WEAK = {"Cheap, deteriorating fundamentals", "Strong long-term, weak short-term catalysts"}   # price downtrend, set by archetype type only
OLD = {"STRONG BUY": 4, "BUY": 3, "HOLD": 2, "SELL": 1, "STRONG SELL": 0}


def snapshot(a):
    built = build(a); info, inc, cf, bs, price = built if len(built) == 5 else (built[0], None, None, None, built[3])
    if inc is None:
        df = pd.DataFrame([{"Category": c, "Score": v} for c, v in (("Profitability", .3), ("Financial health", .2), ("Growth", 0.), ("Valuation", 0.), ("Risk", 0.))])
    else:
        r = np.random.RandomState(abs(hash(a[0])) % 1000); n = 2600
        drift = np.full(n, .0004); drift[-126:] = -.0025 if a[0] in WEAK else .0004
        c = pd.Series(np.exp(np.cumsum(r.normal(drift, .014))), index=pd.bdate_range(end="2026-10-09", periods=n)); c *= price / c.iloc[-1]
        df = ns["build_metrics"](info, pd.DataFrame({"Close": c}), inc, cf, bs, ns["profile_for"](info, info["sector"]))
    cs = df.groupby("Category")["Score"].mean().to_dict()
    return info, inc, bs, price, df, cs


def run():
    rows = []
    for a in ARCH:
        info, inc, bs, price, df, cs = snapshot(a); sec = info["sector"]
        prof = ns["profile_for"](info, sec); bookv = R.use_book_value(info, sec)
        q = float(np.clip((cs.get("Profitability", 0) + cs.get("Growth", 0) + cs.get("Financial health", 0)) / 3, -1, 1))
        d = R.derive(info, price, inc, bs, sec, .0425, info["revenueGrowth"] if not bookv else None, sec in CYC, S2C, sum(prof["over"].get("Operating margin", (.12, .12))) / 2)
        inp = dict(price=price, q=q, ke=d["ke"], hist_g=d["hist_g"], cyc=sec in CYC, dcf_w=(.2 if sec == "Utilities" else .4 if sec in CYC else .6),
                   fwd_eps=info.get("forwardEps") if d["cur_ok"] else None, pe_rng=prof["exit"])
        if bookv: inp["pb"] = dict(bvps=info["bookValue"], roe=info["returnOnEquity"], g=.025)
        elif d["dcf_ok"]: inp["p"] = dict(rev0=d["rev0"], g1=d["g1"], m0=d["m0"], mt=d["mt"], wacc=d["wacc"], tg=d["tg"], tax=.21, s2c=d["s2c"], net_debt=d["nd"] + d["mi"], shares=d["shares"])
        V = R.valuate(inp); vol = float(df.loc[df.Metric == "Annualized volatility", "Value"].iloc[0]) if (df.Metric == "Annualized volatility").any() else .3
        nfl = 2 if d["speculative"] else 0
        # ---- OLD: header ratings from the metric model, overall from R.verdict (two independent pipelines)
        s2, l2 = ns2["analyze"](df, price, vol, info), ns2["analyze_long"](df, price, info, prof); ns2["thesis"](df, info, s2, l2, sec)
        old_ov = None
        if V["val"]:
            sc0 = R.make_scores(cs, V["val"], len(df), nfl, V["disagree"]); tot0, _ = R.overall(sc0, R.DEFAULT_W)
            old_ov = R2.verdict(tot0, V["ret"][5], d["ke"], sc0["Forecast confidence"], V["disagree"])[0]
        # ---- NEW: single source of truth
        s1, l1 = ns["analyze"](df, price, vol, info), ns["analyze_long"](df, price, info, prof); th = ns["thesis"](df, info, s1, l1, sec)
        rec = R.recommend(cs, V, d["ke"], s1["comp"], len(df), nfl, None, .4, th["adj_short"], th["adj_long"]) if V["val"] else None
        rows.append(dict(name=a[0], price=price, old=(OLD[s2["rating"]], OLD[l2["rating"]], R.ORDER.index(old_ov) if old_ov else None), rec=rec,
                         old_labels=(s2["rating"], l2["rating"], old_ov), rated=bool(rec)))
    return rows


def old_violations(o):
    s, l, v = o
    if v is None: return []
    mid = (OLD_TO_NEW := {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})  # legacy SELL/STRONG SELL map onto Avoid/Strong Avoid ordinally
    out = []
    if not min(s, l) <= v <= max(s, l): out.append("overall outside horizon range")
    if min(s, l) >= 3 and v < 3: out.append("both Buy+ but overall below Buy")
    if max(s, l) <= 1 and v > 1: out.append("both Sell- but overall above")
    return out


if __name__ == "__main__":
    rows = run(); table = []
    for r in rows:
        ol = "/".join(x.title() for x in r["old_labels"] if x) if r["old_labels"][2] else "/".join(x.title() for x in r["old_labels"][:2]) + "/n.a."
        ov = old_violations(r["old"]); rec = r["rec"]
        table.append({"Archetype": r["name"], "OLD short/long/overall": ol, "OLD consistent?": "no: " + "; ".join(ov) if ov else "yes",
                      "NEW short": f"{rec['short']['rating']} ({rec['short']['score']:+.2f})" if rec else "not rated", "NEW long": f"{rec['long']['rating']} ({rec['long']['score']:+.2f})" if rec else "-",
                      "NEW overall": f"{rec['overall']['rating']} ({rec['overall']['score']:+.2f})" if rec else "-",
                      "Constraints applied": "; ".join(f"{c['horizon']}: {c['rule']} ({c['from']}→{c['to']})" for c in rec["constraints"]) or "none" if rec else "data issue",
                      "NEW consistent?": "yes" if rec and not R.check_consistency(rec) else ("NO" if rec else "n/a")})
    t = pd.DataFrame(table); pd.set_option("display.width", 250, "display.max_colwidth", 70); print(t.to_string(index=False))
    n_old = sum(1 for r in rows if old_violations(r["old"])); n_new = sum(1 for r in rows if r["rec"] and R.check_consistency(r["rec"]))
    print(f"\nOLD logic: {n_old} of {sum(1 for r in rows if r['old'][2] is not None)} rated archetypes had contradictory ratings.   NEW logic: {n_new} of {sum(r['rated'] for r in rows)}.")
    open(os.path.join(HERE, "rating_matrix.md"), "w").write("# Rating consistency: before vs after (21 synthetic archetypes, not real tickers)\n\n" + t.to_markdown(index=False) + f"\n\nOLD: {n_old} contradictory; NEW: {n_new}.\n")
