"""One pipeline: raw data in -> one result out. The interface and the tests both call analyze(); no page computes its own numbers or labels."""
import numpy as np
import pandas as pd
import research as R, metrics as M, valuation as V, rating as RT, qual as Q
from catalog import CATALOG

clip = lambda x, a=-1.0, b=1.0: float(np.clip(x, a, b))
PLAUS = {"Conservative": .8, "Reasonable": .3, "Aggressive": -.5, "Implausible": -1.0}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _est(df, r, c):
    try:
        return float(df.loc[r, c])
    except Exception:
        return None


def _fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def _top(df, cats, n, best=True):
    d = df[df.Category.isin(cats)].sort_values("Score", ascending=not best).head(n)
    return [f"{r.Metric} {_fmt(r.Value)}" for r in d.itertuples() if (r.Score > .2 if best else r.Score < -.2)]


def analyze(info, hist, inc, cf, bs, est=None, peers=None, ai=None, w_qual=.5, apply_override=False, now=None):
    est = est or {}
    sector, industry = info.get("sector"), info.get("industry")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(hist["Close"].iloc[-1]) if hist is not None and len(hist) else None)
    eg = _est(est.get("rev"), "+1y", "growth") if est.get("rev") is not None else None
    f = V.features(info, inc, cf, bs, eg)
    framework, tags, why = V.classify(f, info, sector, industry)
    fw = RT.FRAMEWORKS[framework]
    prof = M.profile_for(info, sector)
    df = M.build_metrics(info, hist, inc, cf, bs, prof, loss_making=f["loss_making"])
    na = dict(df.attrs.get("na", {}))
    cs = df.groupby("Category")["Score"].mean().to_dict() if len(df) else {}
    quality, health = _mean([cs.get("Profitability"), cs.get("Track record")]), cs.get("Financial health")
    q = _mean([quality, health]) or 0.0
    hurdle = fw["hurdle"]

    scen = V.scenarios(f, info, price, framework, prof["exit"], q)
    reqg, plaus = V.required_growth(f, info, price, framework, prof["exit"], q, hurdle, f["hist_g"]) if scen["ok"] else (None, "n/a")
    mult_rows, ev = V.multiples_table(info, f, framework, price)
    hist_df, hist_stats = V.own_history(info, hist, inc, bs, price)
    peer_df = V.peer_compare(info, f, peers)

    # ---- dimension scores (-1..+1, None = cannot be assessed)
    vitems = []
    if cs.get("Valuation") is not None:
        vitems.append(("Multiples vs sector rules of thumb", cs["Valuation"]))
    key = "P/S" if f["loss_making"] else "P/E"
    if hist_stats and key in hist_stats:
        vitems.append((f"{key} vs own fiscal-year history ({hist_stats[key]['pct']:.0%} percentile)", clip(1 - 2 * hist_stats[key]["pct"])))
    if peer_df is not None:
        prem = peer_df[peer_df.Metric.isin(["Fwd P/E", "EV/EBITDA", "P/S"])]["Premium"]
        if len(prem):
            vitems.append((f"Premium to peer median ({prem.mean():+.0%})", clip(-prem.mean())))
    if plaus != "n/a":
        vitems.append((f"Required growth looks {plaus.lower()}", PLAUS[plaus]))
    rr = None
    if scen["ok"]:
        ev_ann, base_ann, bear_loss = scen["ev_ann"], scen["rows"]["Base"]["ann"], max(0.0, -scen["rows"]["Bear"]["total"])
        rr = clip(.5 * clip((ev_ann - hurdle) / .15) + .3 * clip((base_ann - hurdle) / .15) + .2 * clip(1 - 2 * bear_loss / .6))
    dims = {"Business quality": quality, "Growth": cs.get("Growth"), "Financial health": health, "Valuation": _mean([s for _, s in vitems]),
            "Risk": cs.get("Risk"), "Risk-reward": rr}

    # ---- qualitative view (computed items always; AI items only fill what the data cannot assess)
    qitems = Q.assess_data(inc, cf, bs, info, dict(plaus=plaus, impl=reqg) if reqg is not None else None, loss_making=f["loss_making"], runway=f["runway"])
    if ai:
        for k, it in ai.get("items", {}).items():
            if qitems[k]["score"] is None:
                qitems[k] = it
    qscore, qcov = Q.summarize(qitems)
    override = ai["challenge"] if ai and ai.get("challenge") and apply_override else None

    # ---- coverage, confidence, rating
    have = set(df["Metric"]) if len(df) else set()
    labels = [m for _, m in CATALOG]
    unavailable = [m for m in labels if m not in have and m not in na]
    cov = dict(total=len(CATALOG), evaluated=[m for m in labels if m in have], not_meaningful={m: r for m, r in na.items() if m in labels}, unavailable=unavailable)
    cov_q = len(cov["evaluated"]) / max(len(CATALOG) - len(cov["not_meaningful"]), 1)
    have_est = any(v is not None for v in est.values())
    conf, conf_label = RT.confidence(framework, cov_q, qcov, scen, peer_df is not None, hist_df is not None, have_est)
    raise_ratio = None
    if scen["ok"] and scen["mode"] == "revenue" and f["mc"]:
        raise_ratio = scen["rows"]["Base"]["assump"]["Capital to raise"] / f["mc"]
    rt = RT.rate(framework, dims, scen, conf, dict(score=qscore, coverage=qcov), w_qual, f["runway"], raise_ratio, override)
    trend = est.get("trend")
    rev_rev = None
    try:
        rev_rev = float(trend.loc["+1y", "current"] / trend.loc["+1y", "90daysAgo"] - 1)
    except Exception:
        pass
    nt_score, nt_label = RT.near_term_setup(cs, rev_rev)

    # ---- flags and contradictions
    flags, fm = R.forensic(inc, cf, bs)
    mv = lambda n: (lambda r: float(r.iloc[0]) if len(r) else None)(df.loc[df.Metric == n, "Value"]) if len(df) else None
    cons = R.contradictions(dict(rev_g=info.get("revenueGrowth"), margin_chg=mv("Operating margin change (YoY)"), eps_g=info.get("earningsGrowth"),
                                 oi_g=mv("Operating income growth"), share_chg=fm.get("Share count change (YoY)"), cash_conv=fm.get("Cash conversion (CFO / net income)"),
                                 fwd_pe=info.get("forwardPE"), cyclical="Cyclical" in tags, ps=info.get("priceToSalesTrailing12Months"),
                                 margin_peak=bool(f["m0"] and f["avg_m"] and f["m0"] > f["avg_m"] * 1.15), gross_m=f["gm"], fcf_yield=mv("Free cash flow yield"),
                                 quality=quality, exp_ret=scen.get("ev_ann") if scen["ok"] else None, hurdle=hurdle, analyst=cs.get("Analyst"), rev_trend=rev_rev))

    # ---- deterministic narrative pieces (every number below is computed above)
    name = info.get("shortName") or info.get("longName") or ""
    mc = f["mc"]
    th = [f"{name} is classified as a {fw['label'].lower()}" + (f" ({'; '.join(tags)})" if tags else "") + f". {why[0]}"]
    if mc:
        th.append(f"At ${price:,.2f} the market values it at ${mc / 1e9:,.1f}B" + (f" (enterprise value ${ev / 1e9:,.1f}B)." if ev else "."))
    if scen["ok"]:
        th.append(f"Against a {hurdle:.0%} required return, the base scenario earns {scen['rows']['Base']['ann']:+.1%} a year over {V.H} years and the probability-weighted estimate "
                  f"{scen['ev_ann']:+.1%}, with a bear-case loss of {max(0.0, -scen['rows']['Bear']['total']):.0%}.")
    if reqg is not None:
        th.append(f"Earning the required return needs about {reqg:.0%} year-1 growth, which looks {plaus.lower()}" + (f" against {f['hist_g']:.0%} historical growth." if f["hist_g"] is not None else "."))
    pos = [d for d, s in dims.items() if s is not None and s > .2]
    neg = [d for d, s in dims.items() if s is not None and s < -.1]
    own = [f"{d} ({dims[d]:+.2f}): " + ("; ".join(_top(df, {"Business quality": ["Profitability", "Track record"], "Growth": ["Growth"], "Financial health": ["Financial health"],
                                                            "Valuation": ["Valuation"], "Risk": ["Risk"]}.get(d, []), 2)) or "see scenarios") for d in pos]
    against = [f"{d} ({dims[d]:+.2f}): " + ("; ".join(_top(df, {"Business quality": ["Profitability", "Track record"], "Growth": ["Growth"], "Financial health": ["Financial health"],
                                                                "Valuation": ["Valuation"], "Risk": ["Risk"]}.get(d, []), 2, best=False)) or "see scenarios") for d in neg]
    against += [c["detail"].capitalize() for c in rt["constraints"]] + [f"{x[1]}: {x[2]}" for x in flags[:2]] + cons[:2]
    if scen["ok"]:
        against.append(f"Bear case: {scen['rows']['Bear']['total']:+.0%} versus today's price (probability {scen['rows']['Bear']['prob']:.0%}, an estimate).")
    changes = []
    if scen["ok"]:
        entry = scen["rows"]["Base"]["price_h"] / (1 + hurdle) ** V.H
        changes.append(f"A price near ${entry:,.2f} would make the base case earn the {hurdle:.0%} required return" + (" (current price is above that)." if price > entry else " (current price is already below that)."))
    if reqg is not None:
        changes.append(f"Growth running clearly below about {reqg:.0%} would undercut the base case; clearly above it would strengthen it.")
    if f["runway"] is not None:
        changes.append(f"Funding: cash covers about {f['runway']:.1f} years of burn; a financing on poor terms would raise dilution, and a path to break-even would lower it.")
    changes += [w for w in ("Estimate revisions turning negative would weaken the near-term setup." if nt_label == "Favorable" else "") if w]
    ctx = dict(info=info, hist=hist, inc=inc, cf=cf, bs=bs, est=est, price=price, name=name, f=f, framework=framework, fw=fw, tags=tags, why=why, df=df, cs=cs, dims=dims,
               vitems=vitems, scen=scen, reqg=reqg, plaus=plaus, mult_rows=mult_rows, ev=ev, hist_df=hist_df, hist_stats=hist_stats, peer_df=peer_df,
               qitems=qitems, qscore=qscore, qcov=qcov, override=override, rating=rt, conf=conf, conf_label=conf_label, cov=cov, flags=flags, fm=fm, contradictions=cons,
               near_term=(nt_score, nt_label), thesis=" ".join(th), own=own, against=against, changes=changes, hurdle=hurdle, loss_making=f["loss_making"])
    ctx["questions"] = research_questions(ctx)
    return ctx


def research_questions(c):
    """The ten business questions. Each answer is labelled with where it comes from; nothing is invented."""
    f, info, s = c["f"], c["info"], c["scen"]
    qi = c["qitems"]
    ev_ = lambda k: qi[k]["evidence"] if qi[k]["score"] is not None else None
    rows = [("What does the company do and how does it make money?", (info.get("longBusinessSummary") or "")[:700] or None, "Reported (Yahoo company description)"),
            ("Key products, customers and revenue drivers", None, "Not available from free structured data; needs filings text (AI review can summarise what it is given)"),
            ("Competitive advantage", " ".join(x for x in (ev_("pricing_power"), ev_("moat_economics")) if x) or None, "Computed (financial footprint only; moat type needs judgment)"),
            ("How large could the business become?", (f"Base scenario revenue in {V.H} years: ${s['rows']['Base']['assump']['Revenue at horizon'] / 1e9:,.2f}B vs ${f['rev0'] / 1e9:,.2f}B now. Addressable-market size is not available." if s["ok"] and s["mode"] == "revenue" and f["rev0"] else None), "Estimate (scenario assumption)"),
            ("Structural trends behind growth", None, "Not available from free data; AI review or your own view"),
            ("Evidence for the growth thesis", (f"Blended growth {f['g']:.0%} (analyst estimate {f['est_growth'] if f['est_growth'] is None else format(f['est_growth'], '.0%')}, TTM {info.get('revenueGrowth') if info.get('revenueGrowth') is None else format(info.get('revenueGrowth'), '.0%')}, 3-yr history {f['hist_g'] if f['hist_g'] is None else format(f['hist_g'], '.0%')})." if f["g"] is not None else None), "Computed"),
            ("What must happen for the stock to become much more valuable?", (f"Year-1 growth of about {c['reqg']:.0%} (fading) is needed to earn the {c['hurdle']:.0%} required return; the bull case assumes {s['rows']['Bull']['assump'].get('Revenue growth, year 1', s['rows']['Bull']['assump'].get('EPS growth, year 2', 0)):.0%}." if c["reqg"] is not None and s["ok"] else None), "Estimate"),
            ("What could prevent it?", "; ".join(c["against"][:4]) or None, "Computed flags and scenario risks"),
            ("How much success is already priced in?", (f"Required growth looks {c['plaus'].lower()}" + (f"; trailing P/E sits at the {c['hist_stats']['P/E']['pct']:.0%} percentile of its own fiscal-year history." if c["hist_stats"] and "P/E" in c["hist_stats"] else ".")) if c["plaus"] != "n/a" else None, "Computed"),
            ("Return versus downside at today's price", (f"Base {s['rows']['Base']['ann']:+.1%}/yr, probability-weighted {s['ev_ann']:+.1%}/yr (estimated probabilities), bear case {s['rows']['Bear']['total']:+.0%} in total." if s["ok"] else None), "Estimate")]
    return [dict(Question=q, Answer=a or "Not available: " + (s.get("reason") or "insufficient free data"), Source=src if a else "Not available") for q, a, src in rows]
