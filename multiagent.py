from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, TypedDict, Literal, List
from ollama_functions import start_ollama

from prompts import load_prompt_from_json
from cloudflare import *
from auxiliary_ontology_functions import *

from collections import deque
from ollama_functions import is_model_downloaded

import re
import json

# ---------------------------------
# Queue to keep track of SHACL shapes history
# ---------------------------------
class _ShaclHistoryQueue:
    def __init__(self, maxlen=10, shacl_prefixes=""):
        self.queue = deque(maxlen=maxlen)
        self.shacl_prefixes = shacl_prefixes.strip()

    def add(self, shape: str):
        self.queue.append(shape)

    def get_all(self):
        return list(self.queue)

    def to_string(self):
        content = "\n".join(self.queue)
        return f"{self.shacl_prefixes}\n\n{content}" if self.queue else self.shacl_prefixes

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"_ShaclHistoryQueue({list(self.queue)}, maxlen={self.queue.maxlen})"


# ---------------------------------
# Agent state definition
# ---------------------------------
class _AgentState(TypedDict):
    property: str
    new_queries: List[str]
    ontology_info: str
    rag: str
    is_complete: Literal["yes", "no"]
    property_shapes: str
    shacl_history: _ShaclHistoryQueue
    shacl_prefixes: str
    node_shapes: Dict[str, List[str]]
    iterations: int


# ---------------------------------
# Ontology Agent:
#   - Retrieves ontology details for properties and classes
# ---------------------------------
def _ontology_agent(state: _AgentState, g) -> _AgentState:
    subjects = state["new_queries"]
    ontology_info = state["ontology_info"]

    if state["iterations"] == 0:
        # First iteration: focus on the property itself
        property = subjects[0]
        property_info = get_info_by_name(g, property)
        if property_info is None:
            subjects.remove(property)
            return
        ontology_info += f"# {property}\n{property_info}\n\n"

        property_domain = get_property_domain(g, property)
        if len(property_domain) > 0:
            ontology_info += f"Domain of {property}: {property_domain}\n\n"
        for owl_class in property_domain:
            owl_class_info = get_info_by_name(g, owl_class)
            if owl_class_info is None:
                continue
            ontology_info += f"## {owl_class}\n{owl_class_info}\n\n"
        subjects.remove(property)
    else:
        # Subsequent iterations: expand on new subjects
        for subject in subjects:
            subject_info = get_info_by_name(g, subject)
            if subject_info is None:
                subjects.remove(subject)
                continue
            ontology_info += f"# {subject}\n{subject_info}\n\n"
            subjects.remove(subject)

    return {**state, "ontology_info": ontology_info, "new_queries": subjects}


# ---------------------------------
# Retrieval-Augmented Generation Agent:
#   - Retrieves relevant documents for context
# ---------------------------------
def _rag_agent(state: _AgentState, g, retriever) -> _AgentState:
    print(f"🔍 RAG Agent: Fetching documents for entity {state['property']}")
    docs = retriever.invoke(get_entity_label_and_description(g, state["property"]))
    context_text = ""
    if docs:
        for doc in docs:
            context_text += doc.decode("utf-8") + "\n"
    return {**state, "rag": context_text.strip()}


# ---------------------------------
# Evaluator Agent:
#   - Checks whether the SHACL generation is complete
# ---------------------------------
def _evaluator_agent(state: _AgentState) -> _AgentState:
    print(f"🔍 Evaluator Agent: Evaluating if SHACL is complete for entity {state['property']} (Iteration {state['iterations']})")

    prompt = load_prompt_from_json(prompt_file, "evaluator_agent")
    response = prompt | model | StrOutputParser()
    result = response.invoke({
        "entity": state["property"],
        "ontology": state["ontology_info"],
        "rag": state["rag"],
    })
    parsed_response = result.strip().lower()

    if parsed_response == "yes":
        result = "yes"
        new_queries = []
    else:
        result = "no"
        # Extract possible new queries from brackets
        matches = re.findall(r'\[([^\]]+)\]', parsed_response)
        if matches:
            new_queries = [item.strip().strip('"') for item in matches[0].split(",") if item.strip()]
        else:
            new_queries = []

    print(f"Evaluator result for {state['property']}: {result} (new_queries: {new_queries}) (Iteration {state['iterations']})")
    return {**state, "new_queries": new_queries, "is_complete": result, "iterations": state["iterations"] + 1}


# ---------------------------------
# SHACL cleaning helper
# ---------------------------------
def _clean_shacl_response(response):
    # Remove turtle code blocks if present
    if "```ttl" in response or "```turtle" in response:
        cleaned = re.sub(r"```(?:ttl|turtle)\s*(.*?)```", r"\1", response, flags=re.DOTALL).strip()
    else:
        cleaned = response.strip()
    return cleaned


# ---------------------------------
# Generator Agent:
#   - Generates SHACL constraints for a property
#   - validates syntax and retries on error
# ---------------------------------
def _generator_agent(state: _AgentState) -> _AgentState:
    max_retries = 10
    attempt = 0
    error_message = None

    while attempt < max_retries:
        print(f"🔍 Generator Agent (attempt {attempt + 1}/{max_retries}): Generating SHACL restrictions for entity {state['property']}")

        if error_message:
            prompt = load_prompt_from_json(prompt_file, "generator_agent_property_with_error")
        else:
            prompt = load_prompt_from_json(prompt_file, "generator_agent_property")

        response = prompt | model | StrOutputParser()
        result = response.invoke({
            "entity": state["property"],
            "ontology_info": state["ontology_info"],
            "rag": state["rag"],
            "shacl_history": state["shacl_history"],
            "error": error_message
        })

        if "SHACL shapes not found" in result:
            return {**state}

        result = _clean_shacl_response(result)

        # Validate Turtle syntax
        try:
            temp_graph = Graph()
            combined_ttl = f"{state['shacl_prefixes']}\n{result}"
            temp_graph.parse(data=combined_ttl, format="turtle")
        except Exception as e:
            error_message = e
            attempt += 1
            continue  # retry
        # if valid, add to history
        state["shacl_history"].add(result)
        property_name, affectedClasses = extract_name_and_class_from_shape(result, state["shacl_prefixes"])
        new_node_shapes = update_node_shapes(state["node_shapes"], affectedClasses, property_name)

        print(f"✅ SHACL restrictions for {state['property']}:\n{result}")

        return {
            **state,
            "property_shapes": f"{state['property_shapes']}\n\n{result}",
            "node_shapes": new_node_shapes
        }

    print(f"❌ Reached maximum {max_retries} attempts to generate valid SHACL for entity")
    print(result)
    return {**state}


# ---------------------------------
# Build LangGraph workflow
# ---------------------------------
def _build_graph(g, retriever):
    builder = StateGraph(_AgentState)

    builder.add_node("OntologyAgent", lambda state: _ontology_agent(state, g))
    builder.add_node("RAGAgent", lambda state: _rag_agent(state, g, retriever))
    builder.add_node("EvaluatorAgent", _evaluator_agent)
    builder.add_node("GeneratorAgent", _generator_agent)

    builder.set_entry_point("OntologyAgent")

    # OntologyAgent → RAGAgent on first iteration, EvaluatorAgent otherwise
    builder.add_conditional_edges(
        "OntologyAgent",
        lambda state: "EvaluatorAgent" if state["iterations"] > 0 else "RAGAgent"
    )
    # RAGAgent → EvaluatorAgent
    builder.add_edge("RAGAgent", "EvaluatorAgent")

    # EvaluatorAgent → GeneratorAgent if complete or 3+ iterations, else OntologyAgent
    builder.add_conditional_edges(
        "EvaluatorAgent",
        lambda state: "GeneratorAgent" if (
            state["is_complete"] == "yes" or state["iterations"] >= 3
        ) else "OntologyAgent"
    )
    # GeneratorAgent → END
    builder.add_edge("GeneratorAgent", END)

    return builder.compile()


# ---------------------------------
# Main orchestrator
# ---------------------------------
def run_shacl_generation(g, retriever, model_name, model_temperature, prompting_technique):
    global model
    global prompt_file

    # Choose LLM backend
    if model_name == "gpt":
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=model_temperature
        )
    else:
        if is_model_downloaded("llama3:8b"):
            model = ChatOllama(
                model="llama3:8b",
                temperature=model_temperature
            )
        else:
            raise ValueError("The model 'llama3:8b' is not downloaded in Ollama.")

    # Choose prompting file
    match prompting_technique:
        case "v1":
            prompt_file = "prompts/v1_prompts.json"
        case "few-shot":
            prompt_file = "prompts/few-shot_prompts.json"
        case "cot":
            prompt_file = "prompts/cot_prompts.json"
        case "grounded-citing":
            prompt_file = "prompts/grounded-citing_prompts.json"
        case "all":
            prompt_file = "prompts/all_techniques.json"
        case _:
            prompt_file = "prompts/basic_prompts.json"

    # Build and run LangGraph
    graph = _build_graph(g, retriever)

    shacl_prefixes = """@prefix era: <http://data.europa.eu/949/> .
@prefix era-sh: <http://data.europa.eu/949/shapes/> .
@prefix geosparql: <http://www.opengis.net/ont/geosparql#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix wgs1: <http://www.w3.org/2003/01/geo/wgs84_pos#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix org: <http://www.w3.org/ns/org#> ."""

    property_shapes = shacl_prefixes
    shacl_history = _ShaclHistoryQueue(maxlen=10, shacl_prefixes=shacl_prefixes)
    node_shapes = {}

    # Process each OWL property found in the ontology
    properties = get_owl_properties_with_domain(g)
    for prop in properties:
        print(f"\n\n🔍 Processing property: {prop}")
        state = _AgentState(
            property=prop,
            new_queries=[prop],
            ontology_info="",
            rag="",
            is_complete="no",
            property_shapes=property_shapes,
            shacl_prefixes=shacl_prefixes,
            shacl_history=shacl_history,
            node_shapes=node_shapes,
            iterations=0
        )
        state = graph.invoke(state)
        property_shapes = state["property_shapes"]
        node_shapes = state["node_shapes"]
        shacl_history = state["shacl_history"]

    str_node_shapes = generate_node_shapes_str(node_shapes)
    shacl_completed = f"{shacl_prefixes}\n\n{str_node_shapes}\n\n{property_shapes}"

    print("\n✅ All SHACL constraints generated.")
    return shacl_completed
