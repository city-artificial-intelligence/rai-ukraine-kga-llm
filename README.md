# KG Alignment using LLMs as Oracle

Ontology alignment plays a crucial role in integrating diverse data sources across domains. There is a large plethora of systems that tackle the ontology alignment problem, yet challenges persist in producing highly quality correspondences among a set of input ontologies. Human-in-the-loop during the alignment process is essential in applications requiring very accurate mappings. User involvement is, however, expensive when dealing with large ontologies. In this paper, we explore the feasibility of using Large Language Models (LLM) as an alternative to the domain expert. The use of the LLM focuses only on the validation of the subset of correspondences where an ontology alignment system (i.e., [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher)) is very uncertain. We have conducted an extensive evaluation over several matching tasks of the [Ontology Alignment Evaluation Initiative (OAEI)](http://oaei.ontologymatching.org/), analysing the performance of several state-of-the-art LLMs using different ontology-driven prompt templates. The LLM results are also compared against simulated Oracles with variable error rates.


### Publications
- Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Ernesto Jimenez-Ruiz, Artur d'Avila Garcez. Large Language Models as Oracles for Ontology Alignment. ([arXiv](https://arxiv.org/abs/2508.08500))



### Installation

To install, run:

```
pip install -r requirements.txt
```

You can do it in your venv/conda environment

```
conda activate your_env_name
pip install -r requirements.txt
```

or

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

pyproject.toml is used only for roof linter purposes


### Quick tour

To reproduce the results first of all you need to run the pipeline_with_prompts_processing.ipynb in notebooks folder

You need Gemini API key that should be in your .env file in KG-LLM-Matching directory

```
GEMINI_API_KEY="your_api_key"
```

or OpenAI key

```
OPENAI_API_KEY="your_api_key"
```

#### Model ids

- gpt4o-mini: gpt-4o-mini-2024-07-18
- gemini-2.5-flash-preview: gemini-2.5-flash-preview-04-17
- gemini-2.0-flash: gemini-2.0-flash	
- gemini-2.0-flash-lite: gemini-2.0-flash-lite	
- gemini-1.5-flash: gemini-1.5-flash	


#### Running

You should uncomment/comment parts of pipeline that you want to use like LLM/task/prompt

#### Step-by-step guide:

- First cells contain necesarry imports and helper functions
- Pre-validation: contains health check of prompt and ontology pipeline; verifies working prompt creation
- Pre-processing: creates prompts that already has inserted pairs and relative information. Used for time efficiency as ontology access takes a lot of time and those prompts can be reused and shared for different LLMs
- Run experiments: runs all experiments; here you choose LLM/prompt/benchmark/task
- Analysis: basic analysis that calculates metrics; You need to comment/uncomment/add lines in runs variable to choose which one you want to work with
- Rerun evaluation: just creates analysis for reduced sets of pairs


After running you would basically see results in your run folder:

 - results.csv - for full task
 - results_reduced.csv - for reduced task

Most of analysis is in analyze_results.ipynb

You can also see examples of prompts in prompt_examples.ipynb

### Statistical testing

Statistical significance testing was performed to validate the experimental results. The analysis includes comparison of model performance across different models.

We performed different test for oracle results for 9 tasks.
Tests shows that we are significantly greater then oracle with er 30
and we are significantly close to oracle with er 20

This means each of our improvements over the oracle is statistically significant (for each task separately), so our claims about outperforming oracle baselines were, in fact, correct

Results are in `metrics/satistical_testing`


### Directory structures

```
├── config
│   ├── config.py
│   ├── __pycache__
│   └── runs_vars.py
├── data
│   ├── anatomy
│   ├── bioml-2024
│   ├── largebio
│   └── largebio_small
├── metrics
│   ├── 2025-05-11
│   ├── 2025-05-12
│   ├── satistical_testing/
│   ├── all_runs_metrics.csv
│   ├── all_runs_metrics_reduced.csv
│   ├── OASystems_metrics.csv
│   └── OASystem_with_oracle_f1.csv

├── notebooks
│   ├── analyze_results.ipynb
│   ├── _archive
│   ├── mapping_reduced.ipynb
│   ├── pipeline_with_prompts_preprocessing.ipynb
│   └── prompt_examples.ipynb
├── outputs
│   ├── anatomy
│   ├── _archive
│   ├── best_results
│   ├── bioml-2024
│   └── largebio
├── runs
│   ├── date_of_run
│   ├── all_runs_metrics.csv
│   ├── all_runs_metrics_new.csv
│   ├── all_runs_metrics_new_reduced.csv
│   └── all_runs_metrics_reduced.csv
├── src
│   ├── constants.py
│   ├── evaluate.py
│   ├── formatting.py
│   ├── __init__.py
│   ├── LLM_servers
│   ├── onto_access.py
│   ├── onto_object.py
│   ├── processing.py
│   ├── prompts
│   ├── __pycache__
│   └── utils.py
```

#### Data

The `data` folder contains ontologies and entity pairs required to run the experiments.

Structure is organized as:

`data/benchmark_name/task_name/`

Within each task folder, you will find:
- `oracle_pairs/`: precomputed prompt inputs for querying the oracle.

- `.owl` files: ontology files used in the experiments

- Files like `bioml-2024-ncit-doid-logmap_mappings_to_ask_oracle_user_llm` and `bioml-2024-ncit-doid-logmap_mappings_to_ask_oracle_user_llm_reduced`, which contain pairs of entities used to test with the oracle.

- Some files have been removed due to GitHub’s size limitations. You can find the complete dataset on [Zenodo](https://doi.org/10.5281/zenodo.15394653).

#### Notebooks

The `notebooks` folder contains 4 ipynb files:

- `pipeline_with_prompt_processing.ipynb` is main ipynb file used to run experiments.

- `analyze_results.ipynb` is used to analyze all the experimental data and create visualizations used in the paper.

- `prompt_examples.ipynb` contains example prompts and provides an API for retrieving information about entities.

- `mapping_reduced.ipynb` is a helper notebook used to compute metrics for the reduced set of questions.

#### Src

The `src` folder contains all the code used within the notebooks. It includes helper functions for ontology access, prompt templates, and various utility functions.

#### Runs

The `runs` folder stores all experiment runs that were analyzed in the paper. Each run represents an experiment consisting of one or more tasks, defined by a specific prompt, LLM, and task configuration.

#### Metrics

The `metrics` folder contains combined results of experiments from `runs` folder that were analyzed and used in the paper. `all_runs_metrics.csv` and `all_runs_metrics_reduced.csv` subfolders include metric results based on the full and reduced question sets, respectively, across different prompt, subset, and model configurations. `OASystem_with_oracle_f1.csv` provides metrics result using LLM as an oracle. `OASystems_metrics.csv`' contains various metrics for different OA_Systems evaluations including the LLM-based Oracles.

In the `satistical_testing` subfolder, you can find results of statistical testing performed on the results of the experiments and the scripts used to perform the tests.

Additionally, in subfolders dated `2025-05-11` and `2025-05-12`, you can find result plots for all experiments in the paper, including the performance of the LLM-based Oracle across multiple independent runs, as well as the influence of the system prompt/message.

#### Config

The `config` folder contains `config.py`, which sets root directories for the project, and `runs_vars.py`, which defines datasets names, prompt templates and metadata from previous runs.# KG-LLM-Matching
