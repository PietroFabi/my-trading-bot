"""Render data/backtest/exploration.json as docs/backtest/exploration-report.md (numbers only come from the JSON).
Run from the repo root:  python -m backtest.report"""
import json
import os

from backtest.explore import OUT_JSON, OUT_REPORT


def pct(x, digits=3):
    return "n/a" if x is None or x != x else f"{100 * x:.{digits}f}%"


def num(x, digits=2):
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def render(r: dict) -> str:
    w, s1, s2 = r["window"], r["stage1_choice"], r["stage2_choice"]
    lines = [
        "# Exploration report (Step 9)",
        "",
        f"Exploration window {w['first']} → {w['last']} ({w['sessions']} sessions). Out-of-fold predictions for "
        f"{w['oof_first']} onward ({w['oof_sessions']} sessions), walk-forward by month, expanding window: "
        f"each month is predicted by a model trained only on earlier months. The hold-out (from {w['holdout_start']}) "
        "was not read.",
        "",
        f"Variants tried: {r['variants_tried']['stage1']} model settings + {r['variants_tried']['stage2']} trading rules "
        f"= {r['variants_tried']['total']}. Every one of them is listed below, so the best can be judged against how "
        "many were tried.",
        "",
        "## Stage 1: model level (chosen by out-of-fold information coefficient)",
        "",
        "| Price-move headlines | λ | IC (mean daily Spearman) | t | rows |",
        "|---|---|---|---|---|",
    ]
    for x in r["stage1"]:
        name = "composite benchmark" if x.get("benchmark") else num(x["lam"], 1)
        lines.append(f"| {'excluded' if x['exclude_price_move'] else 'included'} | {name} | {num(x['ic'], 4)} | "
                     f"{num(x['t'])} | {x['rows']} |")
    lines += ["", f"**Chosen:** λ = {s1['lam']}, price-move headlines "
              f"{'excluded' if s1['exclude_price_move'] else 'included'} (IC {num(s1['ic'], 4)}, t {num(s1['t'])}).",
              "", "### IC by month (chosen model vs composite benchmark)", "",
              "| Month | model IC | t | composite IC | t |", "|---|---|---|---|---|"]
    for m, v in r["ic_by_month"]["model"].items():
        c = r["ic_by_month"]["composite"].get(m, {})
        lines.append(f"| {m} | {num(v['ic'], 4)} | {num(v['t'])} | {num(c.get('ic'), 4)} | {num(c.get('t'))} |")

    lines += ["", "## Stage 2: trading rules (chosen by out-of-fold mean net return per trade at the close)", "",
              f"Rules with fewer than {r['min_trades']} out-of-fold trades cannot be chosen.", "",
              "| Entry | Stop (× ATR) | Threshold (× pred. std) | trades | mean net / trade | t | hit rate | zero-trade days |",
              "|---|---|---|---|---|---|---|---|"]
    for x in r["stage2"]:
        lines.append(f"| {x['entry']} | {x['stop'] if x['stop'] is not None else 'none'} | {x['threshold']} | {x['n']} | "
                     f"{pct(x['mean'])} | {num(x['t'])} | {pct(x['hit_rate'], 1)} | {x['zero_trade_days']} |")
    lines += ["", f"**Chosen:** entry {s2['entry']}, stop {s2['stop'] if s2['stop'] is not None else 'none'}, "
              f"threshold {s2['threshold']} × the training predictions' std "
              f"({pct(r['final_model']['threshold_return'], 3)} predicted market-relative return for the frozen model). "
              f"{r['trades_per_day']:.2f} trades per day; {pct(r['share_at_4pct_cap'], 1)} of trades at the 4% cap.",
              "", "## The four claimed candidates (side × entry mode, exit at the close; chosen stop and threshold)", "",
              "| Candidate | trades | mean net / trade | t | hit rate | Sharpe | max drawdown | total return |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in r["claimed_candidates"].items():
        lines.append(f"| {k} | {v['n']} | {pct(v['mean'])} | {num(v['t'])} | {pct(v['hit_rate'], 1)} | "
                     f"{num(v['sharpe'])} | {pct(v['max_drawdown'], 2)} | {pct(v['total_return'], 2)} |")

    lines += ["", "## Chosen rule by side and horizon (other horizons are exploratory: no claim)", ""]
    for side, by_h in r["chosen_rule_by_side_and_horizon"].items():
        lines += [f"### {side}", "",
                  "| Horizon | trades | mean net | t | hit rate | mean gross | Sharpe net | max DD net | Sharpe gross | max DD gross |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for h, v in by_h.items():
            lines.append(f"| {h} | {v['net']['n']} | {pct(v['net']['mean'])} | {num(v['net']['t'])} | "
                         f"{pct(v['net']['hit_rate'], 1)} | {pct(v['gross']['mean'])} | {num(v['curve_net']['sharpe'])} | "
                         f"{pct(v['curve_net']['max_drawdown'], 2)} | {num(v['curve_gross']['sharpe'])} | "
                         f"{pct(v['curve_gross']['max_drawdown'], 2)} |")
        lines.append("")

    dg = r.get("diagnostics")
    if dg:
        lines += ["## Diagnostics (descriptive; nothing above was chosen with them)", "",
                  "Mean market-relative return (open → close) of the gated stocks, out of fold:", "",
                  "| Gate | rows | mean | t | median |", "|---|---|---|---|---|"]
        for k, v in dg["mean_target_by_gate"].items():
            lines.append(f"| {k} | {v['n']} | {pct(v['mean'])} | {num(v['t'])} | {pct(v['median'])} |")
        lines += ["", "Each feature's own IC against the target (out-of-fold gated rows):", "",
                  "| Feature | IC | t |", "|---|---|---|"]
        for k, v in dg["univariate_ic"].items():
            lines.append(f"| {k} | {num(v['ic'], 4)} | {num(v['t'])} |")
        lines.append("")

    b = r["benchmarks"]
    lines += ["## Benchmarks (same out-of-fold sessions)", "",
              f"- SPY open-to-close: mean {pct(b['spy_open_to_close']['mean_daily'])} per day, Sharpe "
              f"{num(b['spy_open_to_close']['sharpe'])}, total {pct(b['spy_open_to_close']['total_return'], 2)}.",
              f"- SPY buy-and-hold: {pct(b['spy_buy_and_hold'], 2)}.", "",
              "## Frozen-model weights (fit on every exploration row; standardised features)", "",
              "| Feature | weight (return per 1 std) |", "|---|---|"]
    fm = r["final_model"]
    for f, c in zip(fm["features"], fm["coef"]):
        lines.append(f"| {f} | {pct(c, 4)} |")
    lines += ["", f"Intercept {pct(fm['intercept'], 4)}; {fm['training_rows']} training rows; λ = {fm['lam']}.", "",
              "Costs: 5 bps per side in every net figure; borrow fee 0 for intraday shorts (assumption). Shortability "
              "uses today's Alpaca easy-to-borrow flags (no history is available).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    with open(OUT_JSON) as f:
        result = json.load(f)
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(render(result))
    print(f"wrote {OUT_REPORT}")


HOLDOUT_JSON = os.path.join("data", "backtest", "holdout.json")
HOLDOUT_REPORT = os.path.join("docs", "backtest", "holdout-report.md")


def render_holdout(r: dict) -> str:
    h = r["holdout"]
    lines = [
        "# Hold-out report (Step 10, evaluated once)",
        "",
        f"Sessions {h['first']} → {h['last']} ({h['sessions']}), read for the first time after the pre-registration "
        f"commit. {h['gated_rows']} gated rows ({h['rows_with_target']} with a measurable outcome).",
        "",
        f"Pass rule (pre-registered): {r['pass_rule']}.",
        "",
        "| Hypothesis | IC | t | trades | trades/day | mean net / trade | t | hit rate | mean gross | Sharpe (net) | max DD | total | passes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("A", "fade"):
        x = r[name]
        n, c = x["eod_net"], x["curve_net"]
        lines.append(f"| {name} | {num(x['ic'], 4)} | {num(x['ic_t'])} | {n['n']} | {num(x['trades_per_day'])} | "
                     f"{pct(n['mean'])} | {num(n['t'])} | {pct(n['hit_rate'], 1)} | {pct(x['eod_gross']['mean'])} | "
                     f"{num(c['sharpe'])} | {pct(c['max_drawdown'], 2)} | {pct(c['total_return'], 2)} | "
                     f"{'yes' if x['passes'] else 'no'} |")
    lines += ["", f"**Goes live (pre-registered decision rule): {r['goes_live']}.**", ""]
    for name in ("A", "fade"):
        lines += [f"## {name} by side and horizon (only the close is claimed)", "",
                  "| Side | Horizon | trades | mean net | t | hit rate |", "|---|---|---|---|---|---|"]
        for side in ("long", "short"):
            for hz, v in r[name]["by_side_and_horizon"][side].items():
                lines.append(f"| {side} | {hz} | {v['net']['n']} | {pct(v['net']['mean'])} | {num(v['net']['t'])} | "
                             f"{pct(v['net']['hit_rate'], 1)} |")
        lines.append("")
    b = r["benchmarks"]
    lines += ["## Benchmarks (same sessions)", "",
              f"- SPY open-to-close: mean {pct(b['spy_open_to_close']['mean_daily'])} per day, total "
              f"{pct(b['spy_open_to_close']['total_return'], 2)}.",
              f"- SPY buy-and-hold: {pct(b['spy_buy_and_hold'], 2)}.", "",
              "Per-trade means weight every trade equally; the Sharpe, drawdown and total use the dollar P&L of "
              "equal-risk sizing (larger positions in calmer stocks), so the two can differ in sign.", ""]
    return "\n".join(lines)
