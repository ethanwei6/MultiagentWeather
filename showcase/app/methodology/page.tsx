"use client";

import { useSyncExternalStore } from "react";
import { ArrowDownToLine, ArrowUpRight } from "lucide-react";
import ModelSpec from "../../components/model-spec";
import evaluation from "../../public/model-evaluation.json";
import results from "../../public/results.json";
import evidence from "../../public/evidence.json";
import "./methodology.css";

const topics = [
  { id: "validation", label: "Validation" },
  { id: "specification", label: "Mathematical specification" },
  { id: "evidence", label: "Evidence & downloads" },
] as const;
const subscribeToTopic = (notify: () => void) => {
  window.addEventListener("hashchange", notify);
  return () => window.removeEventListener("hashchange", notify);
};
const readTopic = () => topics.find(topic => `#${topic.id}` === window.location.hash)?.id ?? "validation";
const defaultTopic = () => "validation" as const;
const fixed = (number: number) => number.toFixed(5);
const count = (number: number) => number.toLocaleString("en-US");
const { weatherEvaluation: weather, marketDiagnostic: market, gates } = evaluation;
const { cohort, metrics } = results;
const scoreGroups = [
  { target: "Observed weather", rows: weather.rows, scores: [
    { model: "Forecast baseline", brier: weather.baseline.brier, loss: weather.baseline.logLoss },
    { model: "Selected model family", brier: weather.candidate.brier, loss: weather.candidate.logLoss },
  ] },
  { target: "Market payout", rows: market.rows, scores: [
    { model: "Forecast baseline", brier: market.baseline.brier, loss: market.baseline.log_loss },
    { model: "Selected model family", brier: market.candidate.brier, loss: market.candidate.log_loss },
    { model: "Market midpoint", brier: market.marketMidpoint.brier, loss: market.marketMidpoint.log_loss },
  ] },
];
const downloads = [
  { label: "Execution cohort", description: `${results.records.length} synthetic positions, metrics and exclusions`, file: "results.json" },
  { label: "Model evaluation", description: "Scores, splits, holdouts and audit limits", file: "model-evaluation.json" },
  { label: "Trade case histories", description: "Selected synthetic trades and example quote snapshots", file: "evidence.json" },
];

function Validation() {
  const bootstrap = weather.cityDateClusterBootstrap;
  return <section aria-label="Validation" className="md-content">
    <div className="md-layout">
      <section className="md-ruled">
        <div className="md-section-head"><h2>Out-of-sample scores</h2><span>Jan–Jun 2026</span></div>
        <p className="md-text">The selected family improved both scores on observed weather. On resolved market payouts, the market midpoint scored best. Lower scores are better within each target.</p>
        <div className="md-scroll"><table className="data-table md-score-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Brier</th><th scope="col">Log loss</th></tr></thead>
          {scoreGroups.map(group => <tbody key={group.target}>
            <tr className="md-group-title"><th scope="rowgroup" colSpan={3}>{group.target} <span>· {count(group.rows)} rows</span></th></tr>
            {group.scores.map(score => <tr key={score.model}><th scope="row">{score.model}</th><td>{fixed(score.brier)}</td><td>{fixed(score.loss)}</td></tr>)}
          </tbody>)}
        </table></div>
        <p className="md-small">Rows repeat forecasts, buckets, decision times, and quote sides. The targets use different populations.</p>
      </section>
      <aside className="md-ruled md-aside">
        <h2>Geographic holdouts</h2>
        <p className="md-text">A group passes when the candidate improves both Brier score and log loss over the forecast baseline. Each test requires a 70% pass rate.</p>
        <dl className="md-gates">{[{ name: "Cities", gate: gates.cityHoldout }, { name: "Regions", gate: gates.regionHoldout }].map(({ name, gate }) => <div key={name}>
          <dt>{name}</dt><dd><strong>{gate.passed} / {gate.total}</strong><span>{(gate.passRate * 100).toFixed(1)}% · Failed</span></dd>
        </div>)}</dl>
        <p className="md-small">The city and region holdouts reuse historical data. Their observations overlap the temporal evaluation.</p>
      </aside>
    </div>
    <section className="md-estimate"><h2>Uncertainty in the weather scores</h2>
      <p className="md-text">Baseline minus candidate Brier score was <strong>{fixed(weather.improvement.brier)}</strong>. Resampling {count(bootstrap.clusters)} city–date clusters {bootstrap.iterations} times gave a 95% interval of <strong>[{bootstrap.brierImprovementInterval.map(fixed).join(", ")}]</strong>.</p>
    </section>
    <div className="md-disclosures">
      <details><summary>Model selection, metrics and temporal splits</summary><div className="md-detail-body">
        <p>Each outer test month used a configuration selected on an earlier inner validation month. The primary estimates use a zero-day embargo. Separate checks used 3- and 7-day embargoes.</p>
        <div className="md-formulas"><span>BS = N⁻¹ Σᵢ (pᵢ − yᵢ)²</span><span>LL = −N⁻¹ Σᵢ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)]</span></div>
        <div className="md-scroll"><table className="data-table"><thead><tr><th scope="col">Test month</th><th scope="col">Train rows</th><th scope="col">Inner validation</th><th scope="col">Test rows</th></tr></thead><tbody>{evaluation.outerSplits.filter(split => split.embargo_days === 0).map(split =>
          <tr key={split.target_month}><th scope="row">{split.target_month}</th><td>{count(split.train_n)}</td><td>{count(split.inner_valid_n)}</td><td>{count(split.test_n)}</td></tr>)}
        </tbody></table></div>
        <p>The archived audit recorded zero train/test city–date overlap and zero unavailable-label rows under its purge rules. Raw-input availability remains an independent verification question.</p>
      </div></details>
      <details><summary>Scope and limitations</summary><div className="md-detail-body"><ul>{evaluation.limitations.map(limitation => <li key={limitation}>{limitation}</li>)}</ul></div></details>
    </div>
    <p className="md-small md-archive-note">The July 7, 2026 report evaluates historical model-family selection. It contains no fixed-S01 P&L backtest. The observed-high adjustment chiefly helps same-day forecasts, while S01 enters one or two days ahead.</p>
  </section>;
}

function Evidence() {
  const implementation = "https://github.com/ethanwei6/MultiagentWeather/blob/d83fb815763bb8df71df842886674f35814a087f/daily_weather_pipeline/weather_model_improvement.py";
  return <section aria-label="Evidence and downloads" className="md-content">
    <div className="md-layout">
      <section className="md-ruled">
        <h2>Synthetic execution fixture</h2>
        <p className="md-text">All trade IDs, times, forecasts, quotes, sizes, fees and payouts in the public demonstration are fictional. The {cohort.fullyReconciledBuyRecords} completed examples supply the P&L chart and trade ledger. Their outcomes were assigned to illustrate both wins and losses.</p>
        <h3>Missing payouts</h3>
        <p className="md-text">The fixture includes {cohort.missingPayoutChainBuyRecords} additional buys with {"$" + fixed(cohort.missingPayoutChainEntryCostUsd)} of entry cost and intentionally absent payouts. Performance statistics use only examples with an assigned outcome.</p>
        <h3>Forecast validation</h3>
        <p className="md-text">The validation tab reports aggregate scores from the July 2026 historical model-family evaluation. The synthetic executions play no part in that evaluation. The equations describe the published model implementation.</p>
        <a className="md-link" href={implementation} target="_blank" rel="noreferrer">Pinned model implementation <ArrowUpRight size={14} aria-hidden="true"/></a>
      </section>
      <aside className="md-ruled md-aside"><h2>Position accounting</h2>
        <p className="md-equation">Π = payout − (shares × fill price) − fee</p>
        <dl className="md-ledger">{[["Entry cost with fees", `$${fixed(metrics.entryCostUsd)}`], ["Trading fees", `$${fixed(metrics.feesUsd)}`], ["Reconciled payout", `$${fixed(metrics.payoutUsd)}`], ["City–date events", cohort.uniqueCityDates]].map(([label, amount]) => <div key={label}><dt>{label}</dt><dd>{amount}</dd></div>)}</dl>
        <p className="md-small">The performance chart groups recorded cash P&L by reconciliation date. Entry cost measures turnover. The figures exclude gas, taxes and unrecorded costs.</p>
      </aside>
    </div>
    <section className="md-downloads"><h2>Download the source data</h2>{downloads.map(item => <a href={`/${item.file}`} key={item.file} download><strong>{item.label}</strong><span>{item.description}</span><code>{item.file}</code><ArrowDownToLine size={16} aria-hidden="true"/></a>)}</section>
    <p className="md-small">{evidence.cases.length} synthetic trades include sparse quote histories. A fixture payout is specified separately from its last quote. The downloaded JSON files are the same inputs used by this interface.</p>
  </section>;
}

export default function MethodologyPage() {
  const topic = useSyncExternalStore(subscribeToTopic, readTopic, defaultTopic);
  const content = { validation: <Validation/>, specification: <ModelSpec/>, evidence: <Evidence/> };
  return <div className="md-page">
    <header className="page-heading"><div><h1>Methodology</h1><p>Historical model-family validation, the frozen S01 equations, and synthetic execution fixtures.</p></div></header>
    <nav className="md-tabs" aria-label="Methodology topics">{topics.map(item => <a key={item.id} href={`#${item.id}`} aria-current={topic === item.id ? "page" : undefined}>{item.label}</a>)}</nav>
    {content[topic]}
  </div>;
}
