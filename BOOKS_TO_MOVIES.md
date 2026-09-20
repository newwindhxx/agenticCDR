# Books to Movies AgentCF++

This fork adds a Linux-compatible, deterministic pipeline under `agentic_cdr/`.
Books is the source domain and Movies and TV is the target domain. The original
research scripts remain unchanged for reference.

## Setup and smoke run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_amazon2023.py --config configs/books_to_movies.yaml
python scripts/prepare_pair.py --config configs/books_to_movies.yaml

export DEEPSEEK_API_KEY='your-key'

# One user and one training interaction: verify the live API and full pipeline first.
python scripts/estimate_run.py --config configs/books_to_movies.yaml --profile trial
python scripts/train_agentcfpp.py --config configs/books_to_movies.yaml --profile trial --resume
python scripts/evaluate_agentcfpp.py --config configs/books_to_movies.yaml --profile trial --split validation

# The fixed 10-user smoke experiment starts in a separate run.
python scripts/estimate_run.py --config configs/books_to_movies.yaml --profile smoke
python scripts/train_agentcfpp.py --config configs/books_to_movies.yaml --profile smoke --resume
python scripts/evaluate_agentcfpp.py --config configs/books_to_movies.yaml --profile smoke --split validation
python scripts/evaluate_agentcfpp.py --config configs/books_to_movies.yaml --profile smoke --split test
```

## Full group-memory run

```bash
python scripts/estimate_run.py --config configs/books_to_movies.yaml --profile full
python scripts/train_agentcfpp.py --config configs/books_to_movies.yaml --profile full --resume
python scripts/evaluate_agentcfpp.py --config configs/books_to_movies.yaml --profile full --split test
python scripts/build_group_memory.py --config configs/books_to_movies.yaml --profile full
python scripts/evaluate_agentcfpp.py --config configs/books_to_movies.yaml --profile full --split test --use-group-memory
```

The downloader resumes `.part` files. Training resumes from an atomic memory
state, and validated LLM responses are cached in the run directory. Raw data,
processed data, caches, memories, and API credentials are ignored by Git.
