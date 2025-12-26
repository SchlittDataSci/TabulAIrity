# Tabulairity

**TabulAIrity** is a Python framework designed for the extraction of structured tabular data from unstructured text using **Conversational Data Extraction Networks (CDENs)**.

Rather than relying on single-shot prompts, TabulAIrity constructs directed graphs of logic and prompts. This allows for complex, branching extraction workflows where the output of one node determines the trajectory of the conversation, ensuring higher accuracy and context-aware data structuring.

## Core Features

* **Conversational Data Extraction Networks:** Define complex extraction logic using directed graphs (NetworkX). Nodes represent LLM prompts, while edges define logic flows (e.g., `isYes`, `isNo`) based on model responses.
* **Model Agnostic Routing:** Built on top of `litellm`, supporting seamless routing between local models (Ollama) and remote APIs (OpenAI, etc.) via configurable routes.
* **Aggressive Caching:** Implements local file-based caching for LLM queries and web scraping to minimize latency and API costs during development and regression testing.
* **Automated Self-Improvement:** Includes an iterative optimization engine that refines prompts against ground-truth datasets to maximize extraction accuracy.
* **Integrated ETL Tools:** Built-in support for Google Sheets I/O, RSS feed ingestion, and automated translation.

## Modules

### `tabulairity.py` (Core)
The primary engine for the framework.
* **Network Construction:** Converts configuration DataFrames into executable `NetworkX` graphs (`buildChatNet`).
* **Execution:** Traverses the graph (`walkChatNet`), managing state (variables), executing prompts, and handling branching logic.
* **Infrastructure:** Manages model routing, API key injection, and response caching.

### `selfimprovement.py`
An automated prompt engineering module.
* **Iterative Optimization:** Takes a prompt and a test dataset (pandas DataFrame), then iteratively rewrites the prompt to improve performance against a supervisor model or ground truth.
* **Intent Extraction:** Analyzes prompts to ensure rewrites preserve the original user intent.
* **Error Summarization:** Aggregates failure cases to guide the supervisor model in specific improvements.

### `gsheetconnector.py`
I/O utilities for external data sources.
* **Google Sheets:** Read/write capabilities for integrating TabulAIrity with cloud-based spreadsheets.
* **Google Alerts/RSS:** Ingests RSS feeds, cleans HTML content, and prepares text for processing.

### `scrapertools.py`
Lightweight web scraping utilities.
* **Text Extraction:** Fetches URLs and utilizes `BeautifulSoup` to strip boilerplate (scripts, styles, nav) and return clean, sentence-structured text for LLM consumption.

## Configuration

TabulAIrity relies on a specific directory structure for configuration and caching.

### Environment Setup
Create a `config/` directory in the project root containing the following:

1.  **`environment_args.txt`**: Stores API keys.
    ```text
    OPENAI_API_KEY = sk-...
    GEMINI_API_KEY = ...
    ```
2.  **`model_routes.csv`**: Defines available models and endpoints.
    ```csv
    model,route,ip
    gemma3:12b,gemma3:12b,http://localhost:11434
    gpt-4o,gpt-4o,remote
    ```
3.  **`config.txt`**: General runtime settings (e.g., paths to Google Service credentials).

### Caching
The system automatically generates a `TabulAIrityCache/` directory. This stores MD5 hashed responses for every LLM query and scrape request. Clear this directory to force fresh execution.

## Usage Example

### Defining a Network
Networks are defined as DataFrames (or loaded from CSV/Google Sheets) with specific columns: `type` (node/edge), `prompt`, `fx` (logic function), and `persona`.

```python
import tabulairity as tb
import pandas as pd

# 1. Load your network definition
net_df = pd.read_csv('my_extraction_logic.csv')

# 2. Build the graph
chat_net = tb.buildChatNet(net_df)

# 3. Execute the network
# 'vars' can be pre-populated with text to analyze (e.g., {'scraped_text': '...'})
results = tb.walkChatNet(chat_net, varStore={'target_text': raw_text_data})

print(results['final_output_node'])
```

### Automated Prompt Improvement
```python
import selfimprovement as si

# Iterate on a prompt to maximize accuracy against a test dataframe
optimized_history = si.iteratePrompt(
    bestPrompt="Extract the date from [text]",
    bestPersona="You are a data extractor.",
    testDfIn=test_data_df,
    model="gemma3:27b"
)
```

## Dependencies

* `pandas`
* `networkx`
* `litellm`
* `beautifulsoup4`
* `gspread`
* `feedparser`
* `matplotlib`
* `pycountry`
* `langdetect`

Slide Deck:```
[Slide Deck](https://docs.google.com/presentation/d/1A5SZgjTdp4PyHzKldXnDUMtvnwshi2Yv1uGauso26ZQ)
