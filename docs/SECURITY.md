# Security

The public showcase and offline tests require no secrets. Some optional research commands can read credentials for external data providers. Keep those credentials in the local environment. Use synthetic literals in tests.

For a public release, review the staged file list, inspect new data fields and third-party assets, run the research tests and frontend build, and scan the exact proposed tree and Git history. The GitHub workflow uses Gitleaks with redacted findings and read-only repository access. Build outputs, private database files, hosted-site manifests and local credential files are ignored.

If you discover an exposed credential, contact the maintainer privately using a reporting channel that does not publish its value. Revoke or rotate the credential before addressing copies in Git history.
