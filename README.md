# Trading Bot (Archived)

> Deprecated — this bot no longer works. Code is kept for learning, reference, and reuse of utilities. Do not use it for real trading.

## Overview
A Python/Node workspace for Telegram-driven token discovery, basic analytics, and trading utilities. Python handles scraping/parsing and simple persistence; Node provides helper scripts for market data and trade orchestration. Includes experimental ML notebooks/scripts and archived analysis.

## Directory Structure
- `telegramDataAnalysis/` — Telethon scraper, parsers, DB helpers, runner (`main.py`).
- `hopefully/` — Node utilities (`traderManager.js`, `priceGatherer.js`).
- `checkstats/` — Small Python tools for token/stat checks.
- `aitests/` — ML experiments (models `.pkl`, datasets `.csv`).
- `Data_analysis2 copy/goodstuff/` — Archived analysis (read-only).

## Prerequisites
- Python 3.10+ and `pip`
- Node.js 18+ and `npm`
- A Telegram API ID/Hash (if exploring the scraper)

## Quick Start
- Python (per module):
  1) `python -m venv .venv && source .venv/bin/activate`
  2) `pip install -r <module>/requirements.txt`
- Node utilities:
  - `cd hopefully && npm ci && npm start` (runs `traderManager.js`)

## Configuration
Create a `.env` for the scraper:
```
API_ID=<your_api_id>
API_HASH=<your_api_hash>
GROUP1=<source_group_id>
GROUP2=<target_group_id>
```

## Usage
- Scraper: `cd telegramDataAnalysis && python main.py`
- Stats check: `python checkstats/checkstats.py`
- Node tools: `cd hopefully && npm start`

## Status & Disclaimer
The bot logic is outdated and non-functional with current APIs/conditions. This repository is provided “as is” without warranty. Use at your own risk for educational purposes only.

## Contributing
See `AGENTS.md` for style, testing, and PR guidance. Keep changes focused, document behavior/configuration, and never commit secrets or large binaries.
