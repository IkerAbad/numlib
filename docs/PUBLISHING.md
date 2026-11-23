# Publishing `numlib-linalg`

The import package is `numlib`. The PyPI distribution is `numlib-linalg`
because the name `numlib` is already registered by an unrelated number-theory
project (version 0.3.1).

## Why the name changed

- Requested name: `numlib`.
- PyPI status: taken by another project.
- Chosen distribution name: `numlib-linalg`.
- Version: `0.1.0`.
- Import name: `numlib` (unchanged).

## Trusted publishing setup

PyPI must authorize this repository before the publish workflow can upload.
Log in to PyPI and open **Account settings > Publishing > Add a new pending
publisher** with exactly these values:

| Field | Value |
| --- | --- |
| PyPI project name | `numlib-linalg` |
| Owner | `IkerAbad` |
| Repository name | `numlib` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

After saving the pending publisher, run:

```bash
gh workflow run publish.yml --ref main
```

or trigger the workflow from the Actions tab. The workflow builds the sdist
and wheel, requests a short-lived OIDC token from PyPI and uploads them. The
`skip-existing` option makes retries safe.

The first failed attempt (before the pending publisher existed) reported these
claims, which are the values that must match the PyPI form:

```text
repository: IkerAbad/numlib
workflow_ref: IkerAbad/numlib/.github/workflows/publish.yml@refs/heads/main
environment: pypi
```

## Alternative: API token

If trusted publishing cannot be configured, create a PyPI API token scoped to
the `numlib-linalg` project and upload locally:

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

The GitHub release `v0.1.0` also carries the same wheel and sdist as assets, so
the package can be installed before the PyPI project is enabled:

```bash
pip install https://github.com/IkerAbad/numlib/releases/download/v0.1.0/numlib_linalg-0.1.0-py3-none-any.whl
```

## After publishing

Validate in a clean directory and a fresh virtual environment:

```bash
git clone https://github.com/IkerAbad/numlib.git
cd numlib
python3.12 -m venv .venv
source .venv/bin/activate
pip install numlib-linalg==0.1.0
python -c "import numlib; print(numlib.__version__)"
python examples/pca.py
python scripts/check_readme_examples.py
```

Finally, update the install note and badge in `README.md` if the PyPI upload
was enabled after the release was cut.
