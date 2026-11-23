# Publishing `numlib-linalg`

The import package is `numlib`. The PyPI distribution is `numlib-linalg`
because the name `numlib` is already registered by an unrelated number-theory
project (version 0.3.1).

## Name mapping

- PyPI distribution: `numlib-linalg`
- Import module: `numlib`
- Version: `0.1.0`
- Repository: `IkerAbad/numlib`

The names differ on purpose. Users install `numlib-linalg` and then run
`import numlib`.

## Trusted publishing

The PyPI project is configured for GitHub Actions trusted publishing with these
values:

| Field | Value |
| --- | --- |
| PyPI project name | `numlib-linalg` |
| Owner | `IkerAbad` |
| Repository name | `numlib` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

To publish a new version, create the tag and release, then run:

```bash
gh workflow run publish.yml --ref main
```

The workflow builds the sdist and wheel, requests a short-lived OIDC token from
PyPI and uploads them. The `skip-existing` option makes retries safe.

## Alternative: API token

If trusted publishing is unavailable, create a PyPI API token scoped to the
`numlib-linalg` project and upload locally:

```bash
python -m pip install -e ".[publish]"
python -m build
TWINE_USERNAME=__token__ TWINE_PASSWORD=... python -m twine upload dist/*
```

Do not commit the token. The trusted-publishing route is preferred because it
does not require a long-lived secret.

## Local checks before uploading

```bash
python -m build
python -m twine check dist/*
```

## Validating an upload

In a clean directory and a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install numlib-linalg==0.1.0
python -c "import numlib; print(numlib.__version__)"
python -c "from numlib import eigen_qr; print(eigen_qr([[2, 1], [1, 2]]).vectors)"
```

The second command confirms that the distribution name and the import name are
different.
