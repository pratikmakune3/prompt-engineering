# Prompt Engineering - Control the LLM beast

A small, practical collection of Python scripts demonstrating core prompt engineering techniques using LangChain and Groq LLMs. The examples cover few-shot prompting, multi-task prompts, in-context learning, and self-consistency.

## Features

- Few-shot sentiment classification
- Multi-task prompting (e.g., sentiment, language detection)
- In-context learning with custom examples
- Self-consistency via multiple reasoning paths and aggregation

## Prerequisites

- Python 3.9+
- A Groq API key

## Setup

1. Clone this repository.
2. Create and activate a virtual environment (recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows (PowerShell): .venv\Scripts\Activate.ps1
```

3. Install dependencies.

```bash
pip install -U langchain groq langchain-groq python-dotenv
```

4. Configure environment variables. Create a `.env` file at the project root:

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

The scripts load environment variables via `python-dotenv`.

## Usage

### prompt01.py — Few-shot and In-Context Learning

This script demonstrates three techniques:

- Few-shot sentiment classification
- Multi-task prompting (e.g., `sentiment`, `language`)
- In-context learning with custom input/output examples

Run:

```bash
python prompt01.py
```

What it does:

- Uses `llama-3.3-70b-versatile` with temperature 0
- Builds prompts with `langchain.prompts.PromptTemplate`
- Chains prompts with the LLM and invokes them

Key functions:

- `few_shot_sentiment_classification(input_text)`: Classifies sentiment as Positive/Negative/Neutral using few-shot examples.
- `multi_task_few_shot(input_text, task)`: Performs a specified task (e.g., sentiment, language) from few-shot examples.
- `in_context_learning(task_description, examples, input_text)`: Learns a mapping from provided examples (e.g., Pig Latin) and applies it to new input.

### prompt03.py — Self-Consistency

Demonstrates the self-consistency technique by generating multiple reasoning paths and aggregating them into a consistent answer, followed by a reliability evaluation.

Run:

```bash
python prompt03.py
```

What it does:

- Uses `llama-3.1-8b-instant` with temperature 0.3
- `generate_multiple_paths(problem, num_paths)`: Produces diverse reasoning paths
- `aggregate_results(paths)`: Asks the model to analyze paths and output the most consistent answer
- `self_consistency_check(problem, aggregated_result)`: Evaluates the aggregated answer’s reliability
- Includes a batch run over example problems

## Configuration Notes

- Models: `llama-3.3-70b-versatile` (accurate, deterministic) and `llama-3.1-8b-instant` (faster, more exploratory)
- Temperatures are chosen to balance determinism and diversity per technique

## Resources

- Google white paper on prompt engineering: [Link](https://drive.google.com/file/d/1AbaBYbEa_EbPelsT40-vj64L-2IwUJHy/view)
