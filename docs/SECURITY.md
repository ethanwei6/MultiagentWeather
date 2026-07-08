# Security

This project may be used with paid data APIs and trading infrastructure. Treat credentials as production secrets.

## Never Commit

- `.env` files
- API keys
- wallet private keys or seed phrases
- CLOB credentials
- funded wallet addresses tied to private operations
- private agent briefs or memory files
- raw live ledgers
- generated logs or reports that may include operational context

## Local Credentials

Use environment variables or a local `.env` file for credentials. `.env` files are ignored by `.gitignore`.

Do not put real credential values in documentation, tests, fixtures, notebooks, logs, or screenshots.

## Release Gate

Before pushing public changes:

1. Confirm the staged file list contains only intended source, tests, and public docs.
2. Run unit tests.
3. Run a redacted secret scan over the exact tree being pushed.
4. Treat any private-key block, API key, wallet key, seed phrase, or credential-looking assignment as a hard stop until manually cleared.

