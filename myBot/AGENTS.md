# Repository Guidelines

Concise contributor guide for this mixed Python/Node trading and data-analysis project. Follow these conventions to keep changes consistent and easy to review.

## Project Structure & Modules
- `telegramDataAnalysis/` — Telethon scraper, parsers, DB helpers, runner (`main.py`).
- `hopefully/` — Node.js utilities (`traderManager.js`, `priceGatherer.js`).
- `checkstats/` — Small Python tools for token/stat checks.
- `aitests/` — ML experiments and datasets (models: `.pkl`, data: `.csv`).
- `Data_analysis2 copy/goodstuff/` — Archived analysis; do not modify.

## Dev, Build, and Run
- Python per-module setup:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r <module>/requirements.txt`
- Run examples:
  - Scraper: `cd telegramDataAnalysis && python main.py`
  - Stats: `python checkstats/checkstats.py`
- Node utilities:
  - `cd hopefully && npm ci && npm start`

## Code Style & Naming
- Python: PEP8, 4 spaces; `snake_case` for files/functions, `CapWords` for classes.
- JavaScript: 2 spaces; `camelCase` for vars/functions, `PascalCase` for classes.
- Filenames: Python `lowercase_with_underscores.py`; keep Node names per existing pattern (e.g., `traderManager.js`).

## Testing
- Prefer lightweight tests colocated with code.
- Python: `tests/` or `test_<module>.py` using `unittest` or simple assertions; aim for key-path coverage.
- Node: add `npm test` when introducing tests; keep fixtures small and deterministic.

## Commits & Pull Requests
- Use imperative, scoped messages (Conventional Commits encouraged):
  - Example: `feat(telegram): add parser for new alert format`
- PRs must include: summary, behavior changes, run steps, and linked issues. Keep scope focused; note perf or breaking changes.

## Security & Config
- Never commit secrets. `.env` keys for Telegram: `API_ID`, `API_HASH`, `GROUP1`, `GROUP2` (used by `telegramDataAnalysis`).
- Avoid committing large binaries; if unavoidable, document regeneration. Treat `*.db` as local artifacts.

## Agent Notes
This AGENTS.md governs the entire `myBot` repo. Tools and scripts should respect the structure above and avoid modifying archived folders.
