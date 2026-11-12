"use client";

import { useState, type PointerEvent } from "react";
import { ArrowRight } from "lucide-react";
import results from "../public/results.json";
import "./performance.css";

const { metrics, cohort, robustness, dailyRealizedPnl: days } = results;
const largestWinner = results.records.reduce((largest, trade) => trade.pnlUsd > largest.pnlUsd ? trade : largest);
const winnerDay = largestWinner.reconciliationTime.slice(0, 10);
const lastIndex = days.length - 1;
const usd = (n: number) => `$${n.toFixed(2)}`;
const signed = (n: number) => `${n < 0 ? "−" : "+"}${usd(Math.abs(n))}`;
const percent = (n: number, digits = 1) => `${(100 * n).toFixed(digits)}%`;
const dateFormat = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const shortDate = (iso: string) => dateFormat.format(new Date(iso));
const timestamp = (iso: string) => Date.parse(iso);
const start = timestamp(days[0].date) - 86400000;
const span = timestamp(days[lastIndex].date) - start;
const x = (iso: string) => (timestamp(iso) - start) / span * 100;

function PerformanceChart() {
  const [mode, setMode] = useState<"Cumulative" | "Daily">("Cumulative");
  const [index, setIndex] = useState(lastIndex);
  const selected = days[index];
  const value = (day: typeof selected) => mode === "Cumulative" ? day.cumulativePnlUsd : day.incrementalPnlUsd;
  const values = days.map(value);
  const range = Math.max(0, ...values) - Math.min(0, ...values);
  const step = Math.max(1, Math.ceil(range / 5));
  const low = Math.floor(Math.min(0, ...values) / step) * step;
  const high = Math.max(low + step, Math.ceil(Math.max(0, ...values) / step) * step);
  const y = (dollars: number) => (high - dollars) / (high - low) * 100;
  const ticks = Array.from({ length: (high - low) / step + 1 }, (_, i) => high - i * step);
  const stepPath = `M0 ${y(0)} ` + days.map(day => `H${x(day.date)} V${y(value(day))}`).join(" ");
  const projectedX = (day: typeof selected) => mode === "Cumulative" ? x(day.date) : x(day.date) - 1.4;
  const peakX = x(winnerDay);
  const city = largestWinner.city.split(",")[0];
  const inspectPoint = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const target = (event.clientX - box.left) / box.width * 100;
    const nearest = days.reduce((best, day, i) => Math.abs(projectedX(day) - target) < Math.abs(projectedX(days[best]) - target) ? i : best, 0);
    setIndex(nearest);
  };
  return (
    <figure className="pf-figure">
      <div className="pf-chart-heading">
        <h2>Reconciled P&L</h2>
        <div className="segmented" role="group" aria-label="Chart measure">
          {(["Cumulative", "Daily"] as const).map(option => <button type="button" key={option} aria-pressed={mode === option} onClick={() => setMode(option)}>{option}</button>)}
        </div>
      </div>
      <div className="pf-figure-value"><strong>{signed(value(selected))}</strong><span>{mode === "Cumulative" ? "Through" : "On"} {shortDate(selected.date)}<br/>{selected.positionCount} positions reconciled that day</span></div>
      <div className="pf-plot">
        <div aria-hidden="true" className="pf-y-ticks">{ticks.map(tick => <span key={tick} style={{ top: `${y(tick)}%` }}>{tick < 0 ? "−" : ""}${Math.abs(tick)}</span>)}</div>
        <div className="pf-plot-body" role="slider" tabIndex={0} aria-label={`${mode} P&L by reconciliation date`} aria-valuemin={1} aria-valuemax={days.length} aria-valuenow={index + 1} aria-valuetext={`${shortDate(selected.date)}, ${signed(value(selected))}, ${selected.positionCount} reconciliations`}
          onKeyDown={event => {
            const position = { ArrowLeft: index - 1, ArrowDown: index - 1, ArrowRight: index + 1, ArrowUp: index + 1, Home: 0, End: lastIndex }[event.key];
            if (position === undefined) return;
            event.preventDefault();
            setIndex(Math.max(0, Math.min(lastIndex, position)));
          }}
          onPointerMove={inspectPoint} onPointerDown={inspectPoint} onPointerLeave={event => { if (event.pointerType === "mouse") setIndex(lastIndex); }}>
          <svg className="pf-line-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {ticks.map(tick => <line key={tick} x1="0" x2="100" y1={y(tick)} y2={y(tick)} className={tick === 0 ? "pf-zero-line" : "pf-gridline"}/>)}
            <line x1={mode === "Cumulative" ? peakX : peakX - 1.4} x2={mode === "Cumulative" ? peakX : peakX - 1.4} y1="0" y2="100" className="pf-event-line"/>
            {mode === "Cumulative" ? <path d={stepPath} className="pf-step"/> : days.map(day => <rect key={day.date} x={projectedX(day) - 1} y={Math.min(y(value(day)), y(0))} width="2" height={Math.abs(y(value(day)) - y(0))} fill={value(day) < 0 ? "var(--red)" : "var(--accent)"}/>)}
            {index !== lastIndex && <line x1={projectedX(selected)} x2={projectedX(selected)} y1="0" y2="100" className="pf-cursor-line"/>}
          </svg>
          <span className="pf-annotation" aria-hidden="true" style={{ left: `${mode === "Cumulative" ? peakX : peakX - 1.4}%`, transform: peakX < 30 ? "translate(4px, -100%)" : undefined }}>{shortDate(winnerDay)} · {city}</span>
          <i aria-hidden="true" className="pf-point" style={{ left: `${projectedX(selected)}%`, top: `${y(value(selected))}%` }}/>
          <div className="pf-x-ticks" aria-hidden="true">{days.filter((_, i) => i % Math.max(1, Math.ceil(days.length / 4)) === 0 || i === lastIndex).map(day => <span key={day.date} style={{ left: `${projectedX(day)}%` }}>{shortDate(day.date)}</span>)}</div>
        </div>
      </div>
      <figcaption>Synthetic USD cash flows by reconciliation date, after example fees. Gas and taxes are excluded.</figcaption>
      <div className="pf-trade-note">
        <a href={`/trades?trade=${largestWinner.id}`}>{largestWinner.title} <ArrowRight size={14} aria-hidden="true"/></a>
        <p>This position earned <strong>{signed(largestWinner.pnlUsd)}</strong> of the {shortDate(winnerDay)} total. Removing it puts the reconciled cohort at <strong className="negative">{signed(robustness.netPnlExcludingLargestWinnerUsd)}</strong>.</p>
      </div>
    </figure>
  );
}

function ExecutionSummary() {
  const rows = [
    ["City–date events", cohort.uniqueCityDates],
    ["Wins / losses", `${metrics.winningPositions} / ${metrics.losingPositions}`],
    ["Win rate", percent(metrics.winRate, 2)],
    ["Profit factor", metrics.profitFactor?.toFixed(3)],
    ["Trading fees", usd(metrics.feesUsd)],
    ["Entry cost, including fees", usd(metrics.entryCostUsd)],
    ["Reconciled payout", usd(metrics.payoutUsd)],
  ];
  return <aside className="pf-ledger" aria-label="Execution summary">
    <h2>Execution ledger</h2>
    <dl>{rows.map(([label, number]) => <div key={label}><dt>{label}</dt><dd>{number}</dd></div>)}</dl>
    <div className="pf-coverage"><h3>Payout coverage</h3>
      <p><strong>{cohort.fullyReconciledBuyRecords} of {cohort.nonCanaryBuyRecords}</strong> synthetic buys have assigned payouts. The remaining {cohort.missingPayoutChainBuyRecords} cost {usd(cohort.missingPayoutChainEntryCostUsd)} to enter and have unknown P&L in the fixture.</p>
      <a href="/methodology#evidence">Inspect the evidence <ArrowRight size={14} aria-hidden="true"/></a>
    </div>
  </aside>;
}

function Attribution() {
  const [group, setGroup] = useState<"side" | "bucket" | "city">("side");
  const rows = [...results.breakdowns[group]].sort((a, b) => b.netPnlUsd - a.netPnlUsd);
  return <section className="pf-attribution">
    <div className="pf-section-heading"><h2>P&L attribution</h2><select aria-label="Group performance by" value={group} onChange={event => setGroup(event.target.value as typeof group)}><option value="side">By side</option><option value="bucket">By bucket</option><option value="city">By city</option></select></div>
    <div className="pf-table-scroll"><table className="data-table"><thead><tr><th scope="col">{group[0].toUpperCase() + group.slice(1)}</th><th scope="col">Trades</th><th scope="col">Win rate</th><th scope="col">Net P&L</th></tr></thead><tbody>
      {rows.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.count}</td><td>{percent(row.winRate)}</td><td className={row.netPnlUsd >= 0 ? "positive" : "negative"}>{signed(row.netPnlUsd)}</td></tr>)}
    </tbody></table></div>
    <p className="pf-note">Positions are grouped by purchased side, contract bucket, or city and ranked by net P&L after trading fees.</p>
    <a className="pf-text-link" href="/trades">Explore individual trades <ArrowRight size={14} aria-hidden="true"/></a>
  </section>;
}

function Calibration() {
  return <section className="pf-calibration"><h2>Purchased-side calibration</h2>
    <p>The model assigned an average <strong>{percent(metrics.meanModelProbability, 2)}</strong> chance of a payout. The purchased side paid out on <strong>{percent(metrics.winRate, 2)}</strong> of reconciled positions.</p>
    <div className="pf-reliability-wrap"><span className="pf-axis-title">Observed payout frequency</span><svg viewBox="0 0 420 168" role="img" aria-label="Reliability diagram. Mean purchased-side model probability is on the horizontal axis and the observed payout frequency is on the vertical axis. Each marker summarizes one populated probability bin.">
      {[0, .5, 1].map(rate => <g key={rate}><line x1="40" x2="392" y1={136 - 116 * rate} y2={136 - 116 * rate} className="pf-gridline"/><text x="30" y={140 - 116 * rate} textAnchor="end">{percent(rate, 0)}</text></g>)}
      {[.7, .8, .9, 1].map(rate => <text key={rate} x={40 + (rate - .7) / .3 * 352} y="160" textAnchor="middle">{percent(rate, 0)}</text>)}
      <line x1="40" x2="392" y1={136 - 116 * .7} y2="20" className="pf-perfect-line"/>
      {results.calibrationBins.map(bin => <circle key={bin.lower} cx={40 + (bin.meanPredicted - .7) / .3 * 352} cy={136 - 116 * bin.observedWinRate} r={3 + Math.sqrt(bin.count) * .45}><title>{`n=${bin.count}, mean probability ${percent(bin.meanPredicted, 2)}, payouts ${percent(bin.observedWinRate, 2)}`}</title></circle>)}
    </svg><span className="pf-axis-title">Mean model probability</span></div>
    <div className="pf-plot-key"><span><i/>Observed frequency</span><span><b/>Perfect calibration</span></div>
    <details className="pf-calibration-details"><summary>Calibration bins and Brier scores</summary>
      <div className="pf-table-scroll"><table className="data-table"><thead><tr><th scope="col">Model probability</th><th scope="col">Trades</th><th scope="col">Mean p</th><th scope="col">Win rate</th></tr></thead><tbody>{results.calibrationBins.map(bin => <tr key={bin.lower}><th scope="row">{percent(bin.lower, 0)}–{percent(bin.upper, 0)}</th><td>{bin.count}</td><td>{percent(bin.meanPredicted, 2)}</td><td>{percent(bin.observedWinRate, 2)}</td></tr>)}</tbody></table></div>
      <p className="pf-note">Brier score is {metrics.brierScore.toFixed(5)} for the model and {metrics.marketAskBrierScore.toFixed(5)} for the decision-time ask. Both scores describe selected executions across {cohort.uniqueCityDates} shared city–date events. The ask includes the spread.</p>
    </details>
  </section>;
}

export default function Home() {
  return <div className="pf-page">
    <header className="page-heading"><div><h1>Performance</h1><p>Synthetic execution examples for daily high-temperature markets. All prices, trades and P&L on this page are fictional.</p></div></header>
    <div className="pf-lead"><PerformanceChart/><ExecutionSummary/></div>
    <div className="pf-analysis"><Attribution/><Calibration/></div>
    <section className="pf-validation"><div><h2>Out-of-sample validation</h2><p>The model improved on the forecast baseline for weather labels. It missed both geographic holdout thresholds and scored worse than the midpoint on market payouts.</p></div><a href="/methodology#validation">View model audit <ArrowRight size={14} aria-hidden="true"/></a></section>
  </div>;
}
