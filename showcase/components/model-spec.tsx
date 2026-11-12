const PUBLIC_COMMIT = "https://github.com/ethanwei6/MultiagentWeather/blob/d83fb815763bb8df71df842886674f35814a087f/daily_weather_pipeline";
const MODEL_SOURCE = `${PUBLIC_COMMIT}/weather_model_improvement.py`;
const VALIDATION_SOURCE = `${PUBLIC_COMMIT}/weather_model_validation.py`;

const equations = [
  {
    number: "01",
    title: "Empirical forecast-error distribution",
    math: `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Residual equals realized high minus forecast high. A group probability is the fraction of its residuals inside the forecast-adjusted bucket."><mtable columnalign="left"><mtr><mtd><msub><mi>r</mi><mi>i</mi></msub><mo>=</mo><msub><mi>y</mi><mi>i</mi></msub><mo>−</mo><msub><mi>f</mi><mi>i</mi></msub></mtd></mtr><mtr><mtd><msub><mi>q</mi><mi>g</mi></msub><mo>=</mo><mfrac><mn>1</mn><msub><mi>n</mi><mi>g</mi></msub></mfrac><munder><mo>∑</mo><mrow><mi>i</mi><mo>∈</mo><mi>g</mi></mrow></munder><mn>1</mn><mo>{</mo><mi>L</mi><mo>−</mo><mi>f</mi><mo>≤</mo><msub><mi>r</mi><mi>i</mi></msub><mo>≤</mo><mi>U</mi><mo>−</mo><mi>f</mi><mo>}</mo></mtd></mtr></mtable></math>`,
    description: "y is the realized daily high; f is the current forecast; [L,U] is the parsed market bucket in °F. Open-ended bounds extend to infinity. Residuals are deduplicated by forecast ID within each group. This is an empirical distribution, not a fitted Gaussian. Frozen S01 adds zero bucket padding.",
    source: `${MODEL_SOURCE}#L767-L795`,
    sourceLabel: "Residual interval probabilities",
  },
  {
    number: "02",
    title: "Evidence-weighted pooling",
    math: `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Group weight equals base weight times n over n plus kappa, provided the minimum sample count is met. Empirical probability is the normalized weighted group average."><mtable columnalign="left"><mtr><mtd><msub><mi>w</mi><mi>g</mi></msub><mo>=</mo><msub><mi>a</mi><mi>g</mi></msub><mfrac><msub><mi>n</mi><mi>g</mi></msub><mrow><msub><mi>n</mi><mi>g</mi></msub><mo>+</mo><msub><mi>κ</mi><mi>g</mi></msub></mrow></mfrac><mn>1</mn><mo>{</mo><msub><mi>n</mi><mi>g</mi></msub><mo>≥</mo><msub><mi>m</mi><mi>g</mi></msub><mo>}</mo></mtd></mtr><mtr><mtd><msub><mi>p</mi><mtext>emp</mtext></msub><mo>=</mo><mfrac><mrow><msub><mo>∑</mo><mi>g</mi></msub><msub><mi>w</mi><mi>g</mi></msub><msub><mi>q</mi><mi>g</mi></msub></mrow><mrow><msub><mo>∑</mo><mi>g</mi></msub><msub><mi>w</mi><mi>g</mi></msub></mrow></mfrac></mtd></mtr></mtable></math>`,
    description: "Groups overlap; this is a weighted mixture of empirical probabilities, not an independent-evidence product or a recursive Bayesian hierarchy. Groups below their minimum count receive zero weight. If none qualify, the model falls back to the forecast-model baseline p₀.",
    source: `${MODEL_SOURCE}#L869-L930`,
    sourceLabel: "Group definitions and shrinkage weights",
  },
  {
    number: "03",
    title: "Tail smoothing and baseline guards",
    math: `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Guarded probability is the clipped mixture of baseline probability and tail-smoothed empirical probability. Beta is the largest applicable far-tail, city, or region guard."><mtable columnalign="left"><mtr><mtd><msub><mi>p</mi><mtext>guard</mtext></msub><mo>=</mo><mi>C</mi><mo>(</mo><mi>β</mi><msub><mi>p</mi><mn>0</mn></msub><mo>+</mo><mo>(</mo><mn>1</mn><mo>−</mo><mi>β</mi><mo>)</mo><mi>T</mi><mo>(</mo><mi>C</mi><mo>(</mo><msub><mi>p</mi><mtext>emp</mtext></msub><mo>)</mo><mo>,</mo><mi>d</mi><mo>)</mo><mo>)</mo></mtd></mtr><mtr><mtd><mi>β</mi><mo>=</mo><mi>max</mi><mo>{</mo><mn>1</mn><mo>{</mo><mi>d</mi><mo>≥</mo><mn>8</mn><mo>}</mo><mo>,</mo><mn>0.25</mn><mn>1</mn><mo>{</mo><msub><mi>N</mi><mi>c</mi></msub><mo>&lt;</mo><mn>35</mn><mo>}</mo><mo>,</mo><mn>0.70</mn><mn>1</mn><mo>{</mo><msub><mi>N</mi><mi>r</mi></msub><mo>&lt;</mo><mn>35</mn><mo>}</mo><mo>}</mo></mtd></mtr></mtable></math>`,
    description: "d is distance to the nearest finite bucket boundary in °F. N꜀ and Nᵣ are the maximum counts across the corresponding city and region groups. p₀ is the source- and model-weighted forecast baseline. These guard coefficients are the frozen S01 configuration, not every configuration evaluated in the research audit.",
    source: `${MODEL_SOURCE}#L957-L977`,
    sourceLabel: "Baseline guards",
  },
  {
    number: "04",
    title: "Side probability, signal, and position P&L",
    math: `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="YES probability is p. NO probability is one minus p. Gross edge is the purchased-side probability minus the ask. Realized profit is payout minus actual fill cost and recorded fee."><mtable columnalign="left"><mtr><mtd><msub><mi>p</mi><mtext>YES</mtext></msub><mo>=</mo><mi>p</mi><mo>,</mo><mspace width="0.7em"/><msub><mi>p</mi><mtext>NO</mtext></msub><mo>=</mo><mn>1</mn><mo>−</mo><mi>p</mi><mo>,</mo><mspace width="0.7em"/><mi>e</mi><mo>=</mo><msub><mi>p</mi><mtext>side</mtext></msub><mo>−</mo><mi>a</mi></mtd></mtr><mtr><mtd><mi>Π</mi><mo>=</mo><mi>q</mi><mo>(</mo><msub><mi>Y</mi><mtext>side</mtext></msub><mo>−</mo><mi>x</mi><mo>)</mo><mo>−</mo><mi>F</mi></mtd></mtr></mtable></math>`,
    description: "p is the final YES probability after all guards; a is the decision-time ask; x is the confirmed fill price; q is shares; F is the recorded fee; Y is the purchased side’s $0/$1 payout. Gross edge is an estimate before costs. Realized P&L uses the actual fill, not the earlier ask, and excludes gas and taxes.",
    source: null,
    sourceLabel: "Archived S01 signal and fill accounting",
  },
  {
    number: "05",
    title: "Risk across mutually exclusive outcomes",
    math: `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Joint event maximum loss is the largest nonnegative loss over every possible event outcome, summing each position's cost minus its scenario payout."><msub><mi>R</mi><mi>e</mi></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><munder><mi>max</mi><mrow><mi>ω</mi><mo>∈</mo><msub><mi>Ω</mi><mi>e</mi></msub></mrow></munder><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>e</mi></mrow></munder><mo>[</mo><msub><mi>c</mi><mi>j</mi></msub><mo>−</mo><msub><mi>q</mi><mi>j</mi></msub><msub><mi>Y</mi><mi>j</mi></msub><mo>(</mo><mi>ω</mi><mo>)</mo><mo>]</mo><mo>)</mo></math>`,
    description: "cⱼ includes the position’s fee. Ω includes every registered temperature outcome plus an OTHER scenario. S01 targets Rₑ ≤ 2% of reconciled equity. Only venue-minimum accommodation may use an 8% event ceiling; it is not an 8% sizing target. Pending orders reserve their worst adverse fill/no-fill contribution separately.",
    source: null,
    sourceLabel: "Frozen S01 parity engine: joint_event_maximum_loss",
  },
];

const groups = [
  ["City × lead day", "0.120", "180", "35"],
  ["City", "0.096", "220", "35"],
  ["Region × lead day", "0.220", "110", "25"],
  ["Region", "0.180", "140", "25"],
  ["Geographic cell × lead day", "0.160", "120", "20"],
  ["Geographic cell × target month", "0.120", "150", "20"],
  ["Forecast source × model family", "0.100", "160", "30"],
  ["Lead day", "0.080", "180", "30"],
  ["Global", "0.080", "1", "1"],
];

export default function ModelSpec() {
  return <div className="panel model-spec">
    <div className="panel-head"><h2 className="panel-title">S01 empirical residual model</h2></div>
    <p className="muted">The frozen S01 configuration pools historical forecast errors and applies baseline and observed-high guards. It estimates each binary temperature contract separately.</p>
    <div className="math-grid">{equations.map(equation => <article className="math-block" key={equation.number}>
      <h3 className="equation-label">{equation.title}</h3>
      <div className="equation" dangerouslySetInnerHTML={{ __html: equation.math }} />
      <p className="muted">{equation.description}</p>
      {equation.source ? <a className="source-ref" href={equation.source} target="_blank" rel="noreferrer">{equation.sourceLabel} ↗</a> : <span className="source-ref">{equation.sourceLabel}</span>}
    </article>)}</div>
    <div className="split">
      <div><div className="panel-head"><h3 className="panel-title">Pooling parameters</h3></div><div className="table-scroll"><table className="data-table"><thead><tr><th>Group</th><th>Base a</th><th>Shrink κ</th><th>Min n</th></tr></thead><tbody>{groups.map(group => <tr key={group[0]}>{group.map((value, i) => i === 0 ? <th key={i} scope="row">{value}</th> : <td key={i}>{value}</td>)}</tr>)}</tbody></table></div><p className="note">Base weights are normalized after evidence scaling; they do not sum to one. “Geographic cell” is the implementation’s 10° latitude × 30° longitude grid, not a climate classification. Regions are US, Europe, and Asia for this universe.</p><a className="source-ref" href={`${MODEL_SOURCE}#L903-L930`} target="_blank" rel="noreferrer">Pooling implementation ↗</a></div>
      <div><div className="panel-head"><h3 className="panel-title">Guard and entry parameters</h3></div>
        <p><strong>Clipping.</strong> C(z) = min(1 − ε, max(ε, z)), ε = 10⁻⁶. The frozen configuration uses tail parameter δ = 0.006 and slope α = 0.65.</p>
        <p><strong>Tail transform.</strong> For d &lt; 4°F, T leaves the clipped probability unchanged. For 4 ≤ d &lt; 8°F, p &lt; δ becomes δ + 0.65(p − δ); p &gt; 1 − δ becomes 1 − δ + 0.65[p − (1 − δ)]; other probabilities stay unchanged. At d ≥ 8°F, β = 1 selects the baseline entirely. δ is a smoothing threshold, not a hard probability floor.</p>
        <p><strong>Observed-high override.</strong> After the baseline guard, a known high above a finite U sets p = ε. For an upper-tail bucket, a known high at or above L sets p = 1 − ε. If the final high is already known, clipped bucket membership replaces p. Missing or future observations cannot trigger an override. This mainly affects same-day research rows; S01 entries use 1–2 day horizons.</p>
        <p><strong>Signal screens.</strong> p<sub>side</sub> ≥ 0.70 and e ≥ 0.08; exact/range buckets additionally require p<sub>side</sub> ≥ 0.98, a ≥ 0.70, and e ≥ 0.10. Quotes must precede the decision by at most 60 seconds and have a nonnegative spread no greater than $0.10. Timing, venue, depth, cash, and portfolio constraints apply separately.</p>
        <p className="note">The historical runner admitted a training label only after its local target day ended plus a 12-hour availability lag. These source-level rules describe the implementation; the model validation results are a separate family-level evaluation.</p>
        <a className="source-ref" href={`${MODEL_SOURCE}#L1119-L1136`} target="_blank" rel="noreferrer">Tail transform ↗</a>{" · "}<a className="source-ref" href={`${MODEL_SOURCE}#L818-L836`} target="_blank" rel="noreferrer">Observed-high guard ↗</a>{" · "}<a className="source-ref" href={`${VALIDATION_SOURCE}#L1983-L2006`} target="_blank" rel="noreferrer">Forecast baseline ↗</a>
      </div>
    </div>
  </div>;
}
