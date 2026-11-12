# Public data

The frontend bundles three small JSON files under `showcase/public/`. The execution results and trade histories use synthetic examples, including invented prices, fills, dates, fees and audit references. They demonstrate the arithmetic and the interface. The model-evaluation snapshot contains aggregate historical validation results. Its weather and market panels measure different targets on different populations.

The raw forecast archive, quote books, private databases, personal transactions, historical fills, account identifiers and research logs stay outside this repository. Researchers running the Python package supply their own data and obtain the provider rights needed for their use. Tests construct temporary local fixtures.

Before adding a new public dataset, document its source and permissions, review identifiers and metadata, and consider whether timestamps, instruments, prices and quantities together could identify an account. Publish only the reviewed fields.
