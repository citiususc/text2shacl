import argparse
from rdflib import Graph

from rag import load_retriever
from multiagent import run_shacl_generation
from auxiliary_ontology_functions import process_shacl
from ollama_functions import start_ollama

import os
import re


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract SHACL constraints from a PDF or TXT file.")

    parser.add_argument("file", help="Path to the text file to be processed.")
    parser.add_argument("--force_process", action="store_true", help="Force reprocessing of the PDF even if it was processed previously.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "gpt"],
        default="llama",
        help="LLM model to use (options: llama, gpt)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature setting for the LLM (default: 0)."
    )
    parser.add_argument(
        "--prompting_technique",
        type=str,
        choices=["v1", "basic", "few-shot", "cot", "grounded-citing", "all"],
        default="basic",
        help="Prompting technique to use (options: basic, few-shot, cot, grounded-citing, all)."
    )


    args = parser.parse_args()

    # Start ollama if the model is open-source
    if not args.model=="gpt":
        start_ollama()

    # Load the graph
    g = Graph()
    g.parse("ontology.ttl", format="turtle")

    # Load and summarize the PDF document
    retriever = load_retriever(args.file, args.model, args.temperature, args.force_process)

    shacl_constraints = run_shacl_generation(g, retriever, args.model, args.temperature, args.prompting_technique)

    output_file = os.path.splitext(os.path.basename(args.file))[0]
    output_file = re.sub(r'[^a-z0-9-]', '-', output_file.lower())
    output_file = f"output/{output_file}_{args.model}_{args.temperature:.2f}_{args.prompting_technique}.ttl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process_shacl(shacl_constraints, output_file)