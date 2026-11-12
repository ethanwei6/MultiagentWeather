"use client";

import { useState } from "react";
import evidence from "../public/evidence.json";
import "./model-lab.css";

const trade = evidence.cases.find(item => item.city.startsWith("Tokyo") && item.side === "NO")!;
const modelSource = "https://github.com/ethanwei6/MultiagentWeather/blob/d83fb815763bb8df71df842886674f35814a087f/daily_weather_pipeline/weather_model_improvement.py";
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const cents = (value: number) => `${Number((value * 100).toFixed(2))}¢`;
const dollars = (value: number, digits = 4) => `$${value.toFixed(digits)}`;
const edge = `+${(trade.grossEdge * 100).toFixed(2)} pp`;

const nodes = {
  forecast: { label: "Forecast feeds", value: `${trade.sources.length} sources`, lines: [`Station ${trade.station}`, `${trade.leadDays}-day horizon`] },
  history: { label: "Historical residuals", value: "rᵢ = yᵢ − fᵢ", lines: ["Realized high minus", "forecast high"] },
  probability: { label: "Empirical bucket model", value: percent(trade.modelProbability), lines: ["P(NO) after guards", `P(YES) ${percent(1 - trade.modelProbability)}`] },
  quote: { label: "Independent quote", value: `${cents(trade.quotedAsk)} ask`, lines: [`NO bid ${cents(trade.quotedBid)}`] },
  edge: { label: "Gross signal edge", value: edge, lines: ["P(NO) minus NO ask"] },
  entry: { label: "Exact-bucket order intent", value: `Buy ${trade.side}`, lines: ["P ≥ 98% · Ask ≥ 70¢", "Edge ≥ 10 pp"] },
  risk: { label: "City–date risk", value: "2% target", lines: ["8% ceiling for", "venue minimums"] },
  fill: { label: "Confirmed fill", value: cents(trade.fillPrice), lines: [`${trade.shares} shares · fee ${dollars(trade.fee)}`, `Cost ${dollars(trade.cost)}`] },
  payout: { label: "Venue resolution", value: dollars(trade.payout, 2), lines: [`NO paid $1 per share`] },
  net: { label: "Realized net P&L", value: `+${dollars(trade.pnl)}`, lines: ["Payout minus cost"] },
};
type NodeId = keyof typeof nodes;

type DiagramNode = { id: NodeId; x: number; y: number; centered?: boolean };

const wideNodes: DiagramNode[] = [
  { id: "forecast", x: 20, y: 82 }, { id: "history", x: 20, y: 208 },
  { id: "probability", x: 230, y: 145 }, { id: "quote", x: 428, y: 20 },
  { id: "edge", x: 428, y: 145 }, { id: "entry", x: 628, y: 145 },
  { id: "risk", x: 628, y: 302 }, { id: "fill", x: 832, y: 145 },
  { id: "payout", x: 1032, y: 20 }, { id: "net", x: 1032, y: 145 },
];
const narrowNodes: DiagramNode[] = [
  { id: "forecast", x: 16, y: 28 }, { id: "history", x: 220, y: 28 },
  { id: "probability", x: 16, y: 168 }, { id: "quote", x: 220, y: 168 },
  { id: "edge", x: 200, y: 312, centered: true }, { id: "entry", x: 16, y: 436 },
  { id: "risk", x: 220, y: 436 }, { id: "fill", x: 16, y: 580 },
  { id: "payout", x: 220, y: 580 }, { id: "net", x: 200, y: 728, centered: true },
];

function FlowDiagram({ narrow }: { narrow: boolean }) {
  const id = narrow ? "ml-narrow" : "ml-wide";
  const positions = narrow ? narrowNodes : wideNodes;
  const description = `${trade.sources.length} example forecast feeds for Tokyo and historical forecast errors enter the bucket model. The example probability of NO was ${percent(trade.modelProbability)}. The independent ask was ${cents(trade.quotedAsk)}, producing a ${edge} gross edge. The exact-bucket screens and event-risk rules precede execution. ${trade.shares} shares filled at ${cents(trade.fillPrice)}. Cost including fees was ${dollars(trade.cost)}. The venue resolved NO and paid ${dollars(trade.payout, 2)}. Reconciled net P&L was ${dollars(trade.pnl)}.`;
  return (
    <svg className={`ml-diagram ${narrow ? "ml-diagram-narrow" : "ml-diagram-wide"}`} viewBox={narrow ? "0 0 400 806" : "0 0 1212 380"} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>Tokyo forecast, independent quote and execution flow</title>
      <desc id={`${id}-desc`}>{description}</desc>
      <defs>
        <marker id={`${id}-head`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M1 1 L7 4 L1 7" className="ml-arrow-head" />
        </marker>
      </defs>
      {narrow ? (
        <g className="ml-wires">
          <path d="M92 108 V124 H284 V108" />
          <path d="M92 124 V149" markerEnd={`url(#${id}-head)`} />
          <path d="M92 250 V268 H284 V242" />
          <path d="M200 268 V293" markerEnd={`url(#${id}-head)`} />
          <path d="M200 375 V392 H92 V417" markerEnd={`url(#${id}-head)`} />
          <path d="M92 518 V538 H284 V518" />
          <path d="M92 538 V561" markerEnd={`url(#${id}-head)`} />
          <path d="M92 662 V684 H284 V646" />
          <path d="M200 684 V709" markerEnd={`url(#${id}-head)`} />
          {[[92,124], [200,268], [92,538], [200,684]].map(([x,y]) => <circle key={y} cx={x} cy={y} r="3" />)}
        </g>
      ) : (
        <g className="ml-wires">
          <path d="M172 112 H196 V238 H172" />
          <path d="M196 175 H219" markerEnd={`url(#${id}-head)`} />
          <circle cx="196" cy="175" r="3" />
          <path d="M374 175 H417" markerEnd={`url(#${id}-head)`} />
          <path d="M488 82 V124" markerEnd={`url(#${id}-head)`} />
          <path d="M584 175 H617" markerEnd={`url(#${id}-head)`} />
          <path d="M696 280 V233" markerEnd={`url(#${id}-head)`} />
          <path d="M784 175 H821" markerEnd={`url(#${id}-head)`} />
          <path d="M974 175 H1021" markerEnd={`url(#${id}-head)`} />
          <path d="M1104 82 V124" markerEnd={`url(#${id}-head)`} />
        </g>
      )}
      {narrow ? (
        <g className="ml-wire-label">
          <text x="108" y="144">model inputs</text><text x="216" y="288">compare</text>
          <text x="104" y="412">price screen</text><text x="108" y="557">sized order</text>
          <text x="216" y="704">reconcile</text>
        </g>
      ) : (
        <g className="ml-wire-label">
          <text x="500" y="108">ask</text><text x="708" y="260">risk budget</text>
          <text x="995" y="163" textAnchor="middle">cost</text><text x="1116" y="108">payout</text>
        </g>
      )}
      {positions.map(({ id: nodeId, x, y, centered }) => {
        const node = nodes[nodeId];
        return (
          <g key={nodeId} className={`ml-node ml-node-${nodeId}`} transform={`translate(${x},${y})`} textAnchor={centered ? "middle" : "start"}>
            <text className="ml-node-label">{node.label}</text>
            <text className="ml-node-value" y="32">{node.value}</text>
            {node.lines.map((line, index) => <text className="ml-node-note" key={line} y={54 + index * 18}>{line}</text>)}
          </g>
        );
      })}
    </svg>
  );
}

const defaultInputs = { probability: "90", ask: "74", spread: "2", age: "20", lead: "1", bucket: "tail" };
const numericFields = [
  { key: "probability", name: "Side probability", unit: "%", max: 100 },
  { key: "ask", name: "Ask price", unit: "¢", max: 100 },
  { key: "spread", name: "Bid–ask spread", unit: "¢", max: 100 },
  { key: "age", name: "Quote age", unit: "s", max: undefined },
] as const;

function EntryScreener() {
  const [inputs, setInputs] = useState(defaultInputs);
  const complete = numericFields.every(({ key }) => inputs[key].trim() !== "" && Number.isFinite(Number(inputs[key])));
  const probability = Number(inputs.probability);
  const ask = Number(inputs.ask);
  const spread = Number(inputs.spread);
  const age = Number(inputs.age);
  const bounded = inputs.bucket === "bounded";
  const gap = probability - ask;
  const checks = [
    { rule: "Model probability", threshold: `≥ ${bounded ? 98 : 70}%`, pass: probability >= (bounded ? 98 : 70) && probability <= 100 },
    { rule: "Gross edge", threshold: `≥ ${bounded ? 10 : 8} pp`, pass: gap + 1e-12 >= (bounded ? 10 : 8) },
    { rule: "Quoted prices", threshold: bounded ? "Ask ≥ 70¢, valid bid" : "Valid bid and ask", pass: ask >= (bounded ? 70 : 0) && ask <= 100 && ask - spread >= 0 && ask - spread <= ask },
    { rule: "Bid–ask spread", threshold: "0–10¢", pass: spread >= 0 && spread <= 10 },
    { rule: "Quote age", threshold: "0–60 s", pass: age >= 0 && age <= 60 },
    { rule: "Forecast horizon", threshold: "1 or 2 days", pass: inputs.lead === "1" || inputs.lead === "2" },
  ];
  const passes = complete && checks.every(check => check.pass);
  const verdict = !complete ? "Enter all inputs" : passes ? "Passes displayed rules" : "Rejected by displayed rules";
  return (
    <section className="ml-screener" aria-labelledby="ml-screener-title">
      <div className="ml-section-heading">
        <h2 id="ml-screener-title">Test the entry rules</h2>
        <p>Vary a hypothetical market quote or model probability to see which S01 threshold binds.</p>
      </div>
      <div className="ml-screener-grid">
        <fieldset className="ml-inputs">
          <legend>Hypothetical inputs</legend>
          <div className="ml-input-grid">
            {numericFields.map(({ key, name, unit, max }) => (
              <label key={key}>
                <span>{name} <small>({unit})</small></span>
                <input type="number" min="0" max={max} step="any" value={inputs[key]} onChange={event => setInputs({ ...inputs, [key]: event.target.value })} />
              </label>
            ))}
            <label>
              <span>Forecast horizon</span>
              <select value={inputs.lead} onChange={event => setInputs({ ...inputs, lead: event.target.value })}>
                {[0, 1, 2, 3].map(day => <option key={day} value={day}>{day} {day === 1 ? "day" : "days"}</option>)}
              </select>
            </label>
            <label>
              <span>Temperature bucket</span>
              <select value={inputs.bucket} onChange={event => setInputs({ ...inputs, bucket: event.target.value })}>
                <option value="tail">Open-ended</option><option value="bounded">Exact or range</option>
              </select>
            </label>
          </div>
        </fieldset>
        <div className="ml-screener-output">
          <div className="ml-verdict" aria-live="polite">
            <strong className={complete ? (passes ? "ml-positive" : "ml-negative") : ""}>{verdict}</strong>
            <span>{complete ? `${gap > 0 ? "+" : ""}${gap.toFixed(2)} pp edge` : "—"}</span>
          </div>
          <table className="ml-check-table"><thead><tr><th>Rule</th><th>Threshold</th><th>Result</th></tr></thead>
            <tbody>{checks.map(check => <tr key={check.rule}><th scope="row">{check.rule}</th><td>{check.threshold}</td><td className={complete ? (check.pass ? "ml-positive" : "ml-negative") : ""}>{!complete ? "—" : check.pass ? "Pass" : "Reject"}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      <p className="ml-footnote">Execution also depends on available market depth, venue order minimums, cash and joint event risk.</p>
    </section>
  );
}

export default function ModelLab() {
  return (
    <div className="ml-page">
      <figure className="ml-figure" aria-labelledby="ml-figure-title">
        <header className="ml-figure-header">
          <div><p className="ml-case-date">Synthetic Tokyo example · {trade.station} · {trade.eventDate}</p><h2 id="ml-figure-title">{trade.title}</h2></div>
          <a href={`/trades?trade=${trade.id}`} className="ml-trade-link">Inspect the example trade <span aria-hidden="true">↗</span></a>
        </header>
        <div className="ml-graph-surface"><FlowDiagram narrow={false} /><FlowDiagram narrow /></div>
        <figcaption className="ml-caption">
          <p><strong>Signal.</strong> The model assigned {percent(trade.modelProbability)} to NO. At a {cents(trade.quotedAsk)} ask, the estimated edge was {edge} on the contract’s $1 payout scale.</p>
          <p><strong>Outcome.</strong> The fill cost {dollars(trade.cost)} including its {dollars(trade.fee)} fee. NO settled at $1 per share and returned {dollars(trade.payout, 2)}, yielding <strong className="ml-positive">{dollars(trade.pnl)} net</strong>.</p>
        </figcaption>
      </figure>
      <div className="ml-annotations">
        <details><summary>Forecast inputs and probability calculation</summary>
          <div className="ml-expanded">
            <p>The example names {trade.sources.join(", ")} as input feeds. It supplies a synthetic feature timestamp, {trade.featureTime.slice(0, 19).replace("T", " ")} UTC, and a fictional model probability. The example does not recompute residuals from provider forecasts.</p>
            <div className="ml-formula">rᵢ = yᵢ − fᵢ<br />q<sub>g</sub> = fraction of rᵢ in [L − f, U − f]<br />w<sub>g</sub> = a<sub>g</sub> n<sub>g</sub> / (n<sub>g</sub> + κ<sub>g</sub>)<br />p<sub>emp</sub> = Σ w<sub>g</sub>q<sub>g</sub> / Σ w<sub>g</sub></div>
            <p>For a new forecast f and parsed bucket [L, U], the model estimates how often past errors would place the high inside the bucket. Nine overlapping groups pool errors across city, region, geographic grid, source and forecast horizon. Each group must reach its minimum sample count to contribute.</p>
            <p>Frozen S01 smooths extreme estimates and blends toward a weighted forecast baseline when local data are sparse. At 8°F or farther from the nearest bucket boundary it uses the baseline. A timely observed high can rule out an impossible outcome. The final YES probability for Tokyo was {percent(1 - trade.modelProbability)}. Its complement gives the synthetic NO estimate.</p>
            <a href={`${modelSource}#L767-L977`} target="_blank" rel="noreferrer">Read the model implementation ↗</a>
          </div>
        </details>
        <details><summary>Entry constraints, event risk and fill accounting</summary>
          <div className="ml-expanded">
            <p>For exact and range buckets, the purchased-side probability must be at least 98%, the ask at least 70¢, and the gross edge at least 10 percentage points. Both sides of the book must be present, the spread at most 10¢, and the quote at most 60 seconds old at decision time. S01 considers one- and two-day horizons.</p>
            <div className="ml-formula">R<sub>event</sub> = max(0, max<sub>ω</sub> −Π(ω))<br />Position cost = {trade.shares} × {dollars(trade.fillPrice, 3)} + {dollars(trade.fee)} = {dollars(trade.cost)}</div>
            <p>For each possible city–date outcome ω, Π(ω) sums the scenario payouts less the fee-inclusive costs of all event positions. S01 targets a worst-case event loss of 2% of reconciled equity. If the proportional order falls below the venue minimum, the minimum-sized order may use up to 8%. The ledger reserves adverse fill/no-fill exposure for pending orders.</p>
            <p>The Tokyo book showed a {cents(trade.quotedBid)} bid and a {cents(trade.quotedAsk)} ask. The example fill price was {cents(trade.fillPrice)} per share. The rule thresholds describe S01’s policy. Replaying its risk check would require the complete portfolio and equity at entry. Example position P&L excludes gas and taxes.</p>
          </div>
        </details>
      </div>
      <EntryScreener />
    </div>
  );
}
