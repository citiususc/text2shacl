# Automatic Constraint Extraction for Knowledge Graphs Using Large Language Models

This project focuses on the automatic extraction of SHACL constraints from textual guides using Large Language Models (LLMs). By leveraging advanced prompt techniques and language models like LLaMA or GPT, it aims to generate accurate SHACL shapes for knowledge graphs, facilitating ontology validation and ensuring data consistency. The system supports various prompting strategies and provides a multi-agent framework to improve constraint extraction quality.


## Project Structure

- **`chroma/`**  
  Stores the vector embeddings generated from the PDF guide fragments (if you have already processed them).  

- **`content/`**  
  Contains the source guide, both in PDF and in HTML format. The final version for use is `RINF_Application_guide_V1.6.1.html`.  

- **`output/`**  
  Contains all generated SHACL constraints in Turtle (`.ttl`) format. Filenames follow the pattern:  
  `{filename}_{model}_{temperature}_{prompting_technique}.ttl`.  

- **`plots/`**  
  Includes the code needed to generate plots and statistics related to the experiments.  

- **`prompts/`**  
  Holds all prompt templates used during the project, organized in JSON files by prompting technique.  

- **`validation/`**  
  Contains code and data used to validate the generated SHACL constraints.  

- **`auxiliary_ontology_functions.py`**  
  Utility functions for ontology processing.  

- **`cloudflare.py`**  
  Helper functions for interacting with Cloudflare R2.  

- **`era-shapes.ttl`**  
  Gold standard of SHACL shapes that serves as the expected target.  

- **`main.py`**  
  The main entry point to run the project.  

- **`multiagent.py`**  
  Implements a multi-agent system to generate constraints collaboratively.  

- **`ollama_functions.py`**  
  Helper functions to interface with the Ollama server.  

- **`ontology.ttl`**  
  Base ontology used as the starting point to generate constraints.  

- **`preprocess_html.py`**  
  Script to preprocess the HTML file converted from the PDF. (There is no need to run it again since the final HTML has already been generated.)  

- **`prompts.py`**  
  Includes a function to load prompt templates from JSON files.  

- **`rag.py`**  
  Implements the RAG (Retrieval-Augmented Generation) technique to build a document retriever.  

- **`requirements.txt`**  
  Project dependencies.  

- **`run_experiments.sh`**  
  Shell script to execute the entire pipeline, including experiments.  

---

## Requirements

To install all dependencies and get the project running:  

```bash
pip install -r requirements.txt
```

In addition, the following conditions and configurations are required:

- **Running Redis server**  
  An active Redis server is essential for the proper functioning of certain modules in the project.

- **`.env` configuration file**  
  A `.env` file must be created containing the following environment variables, required for interacting with Cloudflare R2:

  - `ACCOUNT_ID`: Cloudflare account identifier.  
  - `R2_ACCESS_KEY`: Cloudflare R2 access key.  
  - `R2_SECRET_KEY`: Cloudflare R2 secret key.  
  - `R2_BUCKET`: Name of the R2 bucket, which must be set to public.  
  - `PUB_URL`: Public URL of the R2 bucket.  

- **Ollama server (for the open-source version)**  
  If using the open-source version of the system, an active Ollama server is required, and the `llama3:8b` model must be downloaded.

- **OpenAI API (optional)**  
  If you prefer to use the OpenAI API, the API key must be added to your shell environment. This can be done by adding the following line to your `~/.bashrc` file:

  ```bash
  export OPENAI_API_KEY=your_openai_api_key_here
  ```

To validate the constraints against specific RDF data, the file `ES.zip_combined-new.nq` is used.  
This file is **not included** in the repository due to security and privacy reasons.  
If you need access to it, please request it by email at:  
📧 **adrian.martinez.balea@rai.usc.es**

## Scripts Usage

### Running the Full Experiment Pipeline

To run the entire experimentation process, use:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Running a Single Extraction Execution

You can also run a single extraction using the main script with the following command-line arguments:

```bash
python3 main.py <file_path> [options]
```

#### Arguments

| Argument               | Description                                                                                 | Default | Options                                      |
|------------------------|---------------------------------------------------------------------------------------------|---------|----------------------------------------------|
| `file`                 | Path to the text or PDF file to be processed.                                              | N/A     | N/A                                          |
| `--force_process`      | Forces reprocessing of the PDF even if it has been processed before.                       | False   | Flag (no value needed)                        |
| `--model`              | LLM model to use for constraint extraction.                                               | `llama` | `llama`, `gpt`                               |
| `--temperature`        | Temperature setting for the LLM, controls randomness in generation.                        | `0`     | Any float value                              |
| `--prompting_technique`| Prompting technique to use for the LLM query.                                             | `basic` | `v1`, `basic`, `few-shot`, `cot`, `grounded-citing`, `all` |


#### Example Usage

```bash
python3 main.py content/RINF_Application_guide_V1.6.1.html --model gpt --temperature 0.5 --prompting_technique few-shot
```




