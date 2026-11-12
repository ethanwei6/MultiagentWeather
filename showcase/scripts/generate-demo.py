"""Generate the public dashboard's fictional execution examples with Python 3.

All inputs below are invented. The program reads no account, market, or ledger
data. Run this file to regenerate results.json and evidence.json. The separate
model-evaluation.json contains aggregate historical research statistics.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from statistics import fmean
import json


PUBLIC = Path(__file__).resolve().parents[1] / "public"
# ID, city, April event day, temperature bucket, type, purchased side,
# assigned probability, quoted ask, fill price, shares, assigned binary outcome.
EXAMPLES = [
    ("demo-tokyo-01", "Tokyo, JP", 5, "18°C", "exact", "NO", .9878, .735, .748, 7.25, 1),
    ("demo-madrid-02", "Madrid, ES", 6, "24°C or higher", "above", "YES", .823, .220, .240, 10.50, 1),
    ("demo-chicago-03", "Chicago, IL", 7, "between 55–56°F", "range", "NO", .994, .805, .825, 6.80, 0),
    ("demo-london-04", "London, GB", 8, "17°C or higher", "above", "NO", .917, .705, .705, 7.90, 1),
    ("demo-seattle-05", "Seattle, WA", 9, "52°F or below", "below", "YES", .734, .365, .360, 5.60, 0),
    ("demo-paris-06", "Paris, FR", 10, "21°C", "exact", "NO", .982, .755, .760, 6.30, 1),
    ("demo-seoul-07", "Seoul, KR", 11, "23°C", "exact", "NO", .991, .795, .790, 7.00, 1),
    ("demo-taipei-08", "Taipei, TW", 12, "28°C or higher", "above", "YES", .963, .405, .415, 6.20, 0),
    ("demo-madrid-09", "Madrid, ES", 13, "19°C", "exact", "NO", .988, .815, .805, 6.75, 1),
    ("demo-tokyo-10", "Tokyo, JP", 14, "22°C or higher", "above", "NO", .901, .705, .710, 9.25, 1),
    ("demo-london-11", "London, GB", 15, "14°C", "exact", "NO", .997, .845, .840, 5.80, 0),
    ("demo-chicago-12", "Chicago, IL", 16, "between 61–62°F", "range", "NO", .983, .740, .745, 5.95, 1),
]
FEE_PER_SHARE = Decimal("0.004")
ARCHIVE_END = "2024-04-20T00:00:00Z"
DEMO_NOTE = "Fictional teaching dataset. All trades, dates, prices, probabilities, fills, and payouts in this file are synthetic."


def stamp(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def decimal(value):
    return Decimal(str(value))


def sum_field(rows, field):
    return sum((decimal(row[field]) for row in rows), Decimal(0))


def metrics(rows):
    wins = [row for row in rows if row["pnlUsd"] > 0]
    losses = [row for row in rows if row["pnlUsd"] < 0]
    profit, loss = sum_field(wins, "pnlUsd"), -sum_field(losses, "pnlUsd")
    return {
        "count": len(rows), "winningPositions": len(wins), "losingPositions": len(losses),
        "winRate": fmean(row["outcome"] for row in rows),
        "netPnlUsd": float(sum_field(rows, "pnlUsd")), "notionalUsd": float(sum_field(rows, "notionalUsd")),
        "feesUsd": float(sum_field(rows, "feeUsd")), "entryCostUsd": float(sum_field(rows, "costUsd")),
        "payoutUsd": float(sum_field(rows, "payoutUsd")), "grossProfitUsd": float(profit),
        "grossLossAbsUsd": float(loss), "profitFactor": float(profit / loss) if loss else None,
        "meanModelProbability": fmean(row["modelProbability"] for row in rows),
        "brierScore": fmean((row["modelProbability"] - row["outcome"])**2 for row in rows),
        "meanQuotedAsk": fmean(row["quotedAsk"] for row in rows),
        "marketAskBrierScore": fmean((row["quotedAsk"] - row["outcome"])**2 for row in rows),
        "meanPnlUsd": float(sum_field(rows, "pnlUsd") / len(rows)),
    }


records, cases = [], []
for number, example in enumerate(EXAMPLES, 1):
    record_id, city, day, bucket, kind, side, p, ask, fill, shares, outcome = example
    lead = 1 if number % 2 else 2
    event = datetime(2024, 4, day, 12, tzinfo=timezone.utc)
    quote = event - timedelta(days=lead)
    age = 6 + number * 1.25
    signal = quote + timedelta(seconds=age + 10)
    execution = signal + timedelta(seconds=23)
    redemption = event + timedelta(days=1, minutes=15)
    settlement = redemption + timedelta(seconds=15)
    base_sequence = 1_000_000 + number * 100
    chain = {key: base_sequence + offset for offset, key in enumerate(
        ["signalSequence", "intentSequence", "fillSequence", "redemptionSequence", "settlementSequence"]
    )}
    qty = decimal(shares)
    notional, fee, payout = qty * decimal(fill), qty * FEE_PER_SHARE, qty * outcome
    station = "SIM-" + city.partition(",")[0][:3].upper()
    record = {
        "id": record_id, "synthetic": True, "city": city, "eventDate": event.date().isoformat(),
        "title": f"Will the highest temperature in {city.partition(',')[0]} be {bucket} on April {day}?",
        "side": side, "bucketType": kind, "station": station, "leadDays": lead,
        "signalTime": stamp(signal), "quoteTime": stamp(quote), "quoteAgeAtEvaluationSeconds": age,
        "fillTime": stamp(execution), "reconciliationTime": stamp(settlement),
        "modelProbability": p, "quotedAsk": ask, "grossEdge": round(p - ask, 6),
        "fillPrice": fill, "shares": shares, "notionalUsd": float(notional), "feeUsd": float(fee),
        "costUsd": float(notional + fee), "payoutUsd": float(payout),
        "pnlUsd": float(payout - notional - fee), "outcome": outcome, "evidence": chain,
    }
    records.append(record)
    if number <= 3:
        history = []
        for i, (hours, ask_shift, p_shift) in enumerate([(-12, -.025, -.08), (0, 0, 0), (4, .01, -.015), (8, .035, .007)]):
            observed_ask = round(ask + ask_shift, 3)
            probability = round(min(.999, max(.001, p + p_shift)), 6)
            sample_sequence = base_sequence if hours == 0 else base_sequence + (i - 1) * 10
            history.append({
                "time": stamp(quote + timedelta(hours=hours)), "ask": observed_ask,
                "bid": round(observed_ask - .015, 3),
                "modelProbability": None if number == 1 and hours == -12 else probability,
                "sequence": sample_sequence,
            })
        cases.append({
            **record, "marketId": "illustration-" + str(number), "strategyId": "synthetic-probability-gap",
            "outcome": "win" if outcome else "loss", "settlementValue": outcome,
            "featureTime": stamp(quote - timedelta(minutes=1)), "settlementTime": stamp(settlement),
            "redemptionTime": stamp(redemption), "quotedBid": round(ask - .015, 3),
            "fee": float(fee), "cost": float(notional + fee), "payout": float(payout),
            "pnl": float(payout - notional - fee), "observations": history,
            "sources": ["Synthetic forecast input A", "Synthetic forecast input B"],
            "evidenceType": "Synthetic signal, execution and payout example",
            "forecastMean": None, "residualSigma": None,
        })


# These two illustrative buys have no assigned outcome or payout.
pending = [
    {"id": "demo-pending-13", "synthetic": True, "city": "Paris, FR", "eventDate": "2024-04-17",
     "fillTime": "2024-04-16T12:01:00Z", "side": "NO", "costUsd": 3.75, "missingEvidence": ["unassigned synthetic payout"]},
    {"id": "demo-pending-14", "synthetic": True, "city": "Seattle, WA", "eventDate": "2024-04-18",
     "fillTime": "2024-04-17T12:01:00Z", "side": "YES", "costUsd": 2.50, "missingEvidence": ["unassigned synthetic payout"]},
]

by_date = defaultdict(list)
for row in records:
    by_date[row["reconciliationTime"][:10]].append(row)
cumulative, count, series = Decimal(0), 0, []
for day, rows in sorted(by_date.items()):
    increment = sum_field(rows, "pnlUsd")
    cumulative += increment
    count += len(rows)
    series.append({"date": day, "positionCount": len(rows), "incrementalPnlUsd": float(increment),
                   "cumulativePnlUsd": float(cumulative), "cumulativePositionCount": count})

bins = []
edges = [.70, .80, .90, .95, .98, .99, 1.0]
for low, high in zip(edges, edges[1:]):
    rows = [r for r in records if r["modelProbability"] >= low and
            (r["modelProbability"] < high or high == 1)]
    bins.append({"lower": low, "upper": high, "upperInclusive": high == 1,
                 "count": len(rows), "meanPredicted": fmean(r["modelProbability"] for r in rows) if rows else None,
                 "observedWinRate": fmean(r["outcome"] for r in rows) if rows else None})

breakdowns = {}
for name, field in [("side", "side"), ("bucket", "bucketType"), ("city", "city")]:
    breakdowns[name] = [{"label": label, **metrics([r for r in records if r[field] == label])}
                        for label in sorted({r[field] for r in records})]

winners = sorted((r for r in records if r["pnlUsd"] > 0), key=lambda r: r["pnlUsd"], reverse=True)
total = sum_field(records, "pnlUsd")
all_count = len(records) + len(pending)
results = {
    "schemaVersion": 1, "project": "DailyWeather public demonstration", "synthetic": True,
    "description": DEMO_NOTE, "strategyId": "synthetic-probability-gap",
    "source": {"description": "Deterministic fictional arithmetic examples.", "archiveEndTime": ARCHIVE_END},
    "cohort": {
        "confirmedFillRecords": all_count, "confirmedBuyRecords": all_count, "confirmedSellRecords": 0,
        "excludedTechnicalCanaryFillRecords": 0, "nonCanaryBuyRecords": all_count,
        "fullyReconciledBuyRecords": len(records), "missingPayoutChainBuyRecords": len(pending),
        "missingPayoutChainEntryCostUsd": float(sum_field(pending, "costUsd")),
        "uniqueCityDates": len({(r["city"], r["eventDate"]) for r in records}),
        "uniqueReconciledEvents": len(by_date),
        "settledCohortFirstFillTime": min(r["fillTime"] for r in records),
        "settledCohortLastFillTime": max(r["fillTime"] for r in records),
        "firstReconciliationTime": min(r["reconciliationTime"] for r in records),
        "lastReconciliationTime": max(r["reconciliationTime"] for r in records),
        "missingEvidenceCounts": {"unassigned synthetic payout": len(pending)},
    },
    "metrics": metrics(records), "dailyRealizedPnl": series, "calibrationBins": bins,
    "robustness": {
        "largestWinnerPnlUsd": winners[0]["pnlUsd"],
        "netPnlExcludingLargestWinnerUsd": float(total - sum_field(winners[:1], "pnlUsd")),
        "netPnlExcludingTopFiveWinnersUsd": float(total - sum_field(winners[:5], "pnlUsd")),
    },
    "breakdowns": breakdowns, "records": records, "excludedRecords": pending,
    "methodology": {
        "cohortDefinition": "Twelve fictional buys have assigned binary payouts. Two additional examples have unassigned synthetic payouts.",
        "outcomeDefinition": "An assigned winning share pays $1. An assigned losing share pays $0.",
        "pnlFormula": "Assigned payout minus shares times fill price minus a synthetic trading fee of $0.004 per share.",
        "profitFactorFormula": "Sum of positive position P&L divided by the absolute sum of negative position P&L.",
        "brierFormula": "Mean squared difference between assigned purchased-side probability and its assigned binary outcome.",
        "marketAskBrierFormula": "The same calculation using the illustrative quoted ask as a probability proxy.",
        "timelineDefinition": "Daily P&L grouped by assigned cash reconciliation date in UTC.",
        "signalTimestampSemantics": "Quote evaluation time, signal journal time, fill time and payout time are separately assigned for this demonstration.",
        "robustnessDefinition": "Subtract the largest one or five assigned profitable positions from the sample total.",
        "limitations": [DEMO_NOTE, "This dataset demonstrates calculations and interface behavior. The separate historical model evaluation uses aggregate research statistics."],
    },
}
evidence = {
    "schemaVersion": 1, "project": "DailyWeather public demonstration", "synthetic": True,
    "asOf": ARCHIVE_END, "sourceDescription": DEMO_NOTE, "cases": cases,
    "chartNote": "Quote history points, gaps and execution prices are assigned fictional examples.",
    "pnlNote": results["methodology"]["pnlFormula"],
    "selectionNote": "Three synthetic examples demonstrate a price change, a large winning payout and a loss.",
}
PUBLIC.mkdir(parents=True, exist_ok=True)
for name, content in [("results", results), ("evidence", evidence)]:
    (PUBLIC / (name + ".json")).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
