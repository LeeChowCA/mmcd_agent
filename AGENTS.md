# Repository Guidelines

## Project Structure & Module Organization
This repository is a minimal Python 3.11 scaffold for the `mmcd-agent` service.

- `main.py`: current executable entry point (`main()`).
- `pyproject.toml`: project metadata and dependency declarations.
- `README.md`: project overview (currently empty; update as features are added).

When new code is introduced, prefer this layout:
- `src/mmcd_agent/`: application modules (LangGraph flows, tools, adapters).
- `tests/`: unit and integration tests mirroring `src/`.
- `docs/`: architecture notes and API contracts.

## Build, Test, and Development Commands
Use Python 3.11+.

- `python main.py`: run the current entry point.
- `python -m venv .venv` then `.venv\Scripts\Activate.ps1`: create/activate local environment.
- `pip install -e .`: install the project in editable mode after dependencies are defined.
- `pytest -q`: run tests (once `pytest` is added).
- `ruff check .` and `ruff format .`: lint and format (recommended baseline tooling).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and explicit type hints for public functions.
- Modules/files: `snake_case.py`.
- Classes: `PascalCase`.
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Keep functions small and side effects isolated (especially for LLM/tool orchestration).

## Testing Guidelines
- Place tests under `tests/` with names like `test_<module>.py`.
- Prefer deterministic unit tests for graph nodes and tool wrappers.
- Add integration tests for external dependencies (DB/vector store/API) behind env-gated flags.
- Target meaningful coverage on orchestration logic before adding new features.

## Commit & Pull Request Guidelines
No historical commit pattern exists yet; use this standard:

- Commit format: `type(scope): short summary` (example: `feat(api): add streaming answer endpoint`).
- Keep commits focused and atomic.
- PRs should include: purpose, key changes, test evidence (`pytest` output), and any config/env changes.
- Link related issues/tasks and include request/response examples for API behavior changes.

## Security & Configuration Tips
- Never commit secrets; use environment variables and a local `.env` file ignored by Git.
- Document required variables in `README.md` as they are introduced.
- Validate and sanitize all external inputs before invoking tools or database queries.
