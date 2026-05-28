# research-on-transformers-activations

## Installation

```bash
uv venv --python 3.11 --python-preference only-managed
source .venv/bin/activate
uv pip install -e .
```

### 3) Hugging Face login

Some models/datasets require authenticated access.

```bash
hf auth login
```

## Experiments

### 1) Run all `experiments/base_evals`, then build pie charts

Option A (recommended): run all base eval scripts with one command.

```bash
bash experiments/run_all_base_evals.sh
```

Option B: run the combined evaluator:

```bash
python3 experiments/base_evals/eval_many.py
```
By default the results of `acta` analysis hooks are stored in `.acta_dump_results`.

Then generate pie charts/statistics:

```bash
python3 experiments/collect_stats.py --output visualization
```

In case something went wrong and requires restart please run
```bash
acta clear
```
to clean up corrupted results that could affect pie chart plotting.

### 2) Run `prompt_size_effect.py`

```bash
python3 experiments/prompt_size_effect.py
```

Optional output override:

```bash
python3 experiments/prompt_size_effect.py --output-json outputs/prompt_size_effect_results.json
```

### 3) Run `experiments/explore_train_params_effect.py`

Default run:

```bash
python3 experiments/explore_train_params_effect.py
```

Run all train options (baseline + each option separately):

```bash
bash experiments/run_train_params_all_options.sh
```
