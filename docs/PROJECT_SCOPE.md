# Project Scope

MultiagentWeather is focused on daily-high temperature market research and tooling.

Core technical surfaces:

- normalize market and weather artifacts into auditable local stores
- validate daily-high weather probability models without label leakage
- compare model outputs against market prices when point-in-time quotes exist
- replay candidate trading rules using point-in-time data
- audit settlement labels and weather-observation assumptions
- run forward-shadow collection before any live deployment decision

The public repository does not include private strategy handoff notes, private prompts, agent memory, live credentials, generated datasets, or live-trading operational records.

## Non-Goals For The Public Repo

- publishing private API credentials
- publishing large historical quote/weather databases
- publishing live ledger artifacts
- presenting private strategy decisions as public financial advice

