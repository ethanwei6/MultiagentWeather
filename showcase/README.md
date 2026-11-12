# DailyWeather research showcase

A local, interactive companion to [MultiagentWeather](../README.md). It connects the weather-model equations to entry decisions, fills, payouts and an accounting of realized P&L. The public trade examples and their performance statistics are **fully synthetic**. The validation page separately presents aggregate results from the July 7, 2026 historical evaluation of the model family.

## Run locally

Use Node.js 22.13 or newer. From this directory:

```sh
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000). For the production version, stop the development server and run:

```sh
npm run build
npm start
```

The showcase runs from checked-in JSON and CSS. It requires no account, database, API credentials, or external weather or market feeds.

## Pages

| Page | What to inspect |
| --- | --- |
| [Performance](app/page.tsx) | Switch between daily and cumulative P&L. Inspect the largest winning position, group attribution by side, city or bucket, and compare predicted probabilities with assigned outcomes. |
| [Model & Trader](components/model-lab.tsx) | Follow a synthetic Tokyo case through the empirical residual model, quote comparison, order screens, risk rules and payout. Change a hypothetical input to test the entry thresholds. |
| [Trades](components/trade-explorer.tsx) | Search positions, filter losses and compare the entry probability with observed example quotes and the fill. A `?trade=<demo-id>` URL opens a specific record. |
| [Methodology](app/methodology/page.tsx) | Compare archived weather and payout scores **within each target**. Review geographic holdouts, open the [mathematical specification](components/model-spec.tsx) and inspect the fixture definitions and downloads. |

## Data

- [results.json](public/results.json) supplies the synthetic trades, cohort accounting, daily P&L and conditional calibration diagnostics.
- [evidence.json](public/evidence.json) adds sparse, fictional quote observations for selected examples. Some model probabilities are deliberately absent.
- [model-evaluation.json](public/model-evaluation.json) contains aggregate historical weather-score and market-payout diagnostics. The candidate improved weather scores, failed both geographic holdout gates and underperformed the market midpoint on payout scores. This evaluation did not run a strategy P&L backtest.

To regenerate the two trade fixtures from their fictional inputs, run `python3 scripts/generate-demo.py` from this directory. The generator is independent of the historical aggregate score file.

The UI reads the JSON directly, so its download links return exactly the displayed inputs. Static TypeScript checks run with `npm run typecheck`.
