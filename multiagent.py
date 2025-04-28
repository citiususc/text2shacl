# === multiagent.py ===

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import TypedDict, Literal
from rdflib import Graph
from rag import split_pdf, summarize, embedding

# --- STATE ---
class AgentState(TypedDict):
    entity: str
    ontology_info: str
    rag: str
    is_complete: Literal["yes", "no"]
    shacl: str
    iterations: int

# --- Load Ontology ---
def load_ontology():
    g = Graph()
    g.parse("https://data-interop.era.europa.eu/era-vocabulary/ontology.ttl", format="turtle")
    query_classes = """
        SELECT ?class ?comment WHERE {
            ?class a owl:Class .
            FILTER(!isBlank(?class))
            OPTIONAL { ?class rdfs:comment ?comment . }
            FILTER (lang(?comment) = "en" || !BOUND(?comment))
        }
    """
    class_info = []
    for row in g.query(query_classes):
        class_uri = row[0]
        comment = row[1] if row[1] else "No description available"
        class_info.append({"name": str(class_uri), "description": str(comment)})
    return class_info

# --- Agents ---
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Ontology Agent
def ontology_agent(state: AgentState) -> AgentState:
    print(f"🔍 Ontology Agent: Fetching ontology info for entity {state['entity']}")
    prompt = f"""
    Given the entity: {state['entity']}, extract its definition and related entities from an ontology. 
    Return a structured textual explanation.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"Ontology Info for {state['entity']}:\n{response.content}")
    return {**state, "ontology_info": response.content}

# RAG Agent
def rag_agent(state: AgentState, retriever) -> AgentState:
    print(f"🔍 RAG Agent: Fetching documents for entity {state['entity']}")
    docs = retriever.invoke(state["entity"])
    context_text = ""
    if docs:
        for doc in docs:
            if hasattr(doc, "text"):
                context_text += doc.text + "\n\n"
    return {**state, "rag": context_text.strip()}

# Evaluator Agent
def evaluator_agent(state: AgentState) -> AgentState:
    print(f"🔍 Evaluator Agent: Evaluating if SHACL is complete for entity {state['entity']} (Iteration {state['iterations']})")
    prompt = f"""
    Based on the following information from the ontology and RAG, is it enough to write SHACL restrictions?
    Ontology:
    {state['ontology_info']}

    RAG:
    {state['rag']}

    Reply only with 'yes' or 'no'.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = "yes" if "yes" in response.content.lower() else "no"
    print(f"Evaluator result for {state['entity']}: {result} (Iteration {state['iterations']})")
    return {**state, "is_complete": result, "iterations": state["iterations"] + 1}

# Generator Agent
shacl_accumulator = []
def generator_agent(state: AgentState) -> AgentState:
    print(f"🔍 Generator Agent: Generating SHACL restrictions for entity {state['entity']}")
    prompt = f"""
    Generate SHACL restrictions for the entity:
    {state['entity']}

    Using the following information:
    Ontology:
    {state['ontology_info']}

    RAG:
    {state['rag']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"SHACL restrictions for {state['entity']}:\n{response.content}")
    shacl_accumulator.append(response.content)
    return {**state, "shacl": response.content}

# --- LangGraph Graph ---
def build_graph(retriever):
    builder = StateGraph(AgentState)

    builder.add_node("OntologyAgent", ontology_agent)
    builder.add_node("RAGAgent", lambda state: rag_agent(state, retriever))
    builder.add_node("EvaluatorAgent", evaluator_agent)
    builder.add_node("GeneratorAgent", generator_agent)

    builder.set_entry_point("OntologyAgent")
    builder.add_edge("OntologyAgent", "RAGAgent")
    builder.add_edge("RAGAgent", "EvaluatorAgent")
    builder.add_conditional_edges(
        "EvaluatorAgent",
        lambda state: "GeneratorAgent" if state["is_complete"] == "yes" or state["iterations"] >= 3 else "OntologyAgent"
    )
    builder.add_edge("GeneratorAgent", END)

    return builder.compile()

# --- Main Execution ---
def run_shacl_generation(retriever):
    entities = load_ontology()
    graph = build_graph(retriever)

    for entity in entities:
        print(f"\n\n🔍 Processing entity: {entity['name']}")
        state = AgentState(
            entity=entity["name"],
            ontology_info="",
            rag="",
            is_complete="no",
            shacl="",
            iterations=0
        )
        graph.invoke(state)

    print("\n✅ All SHACL constraints generated. Total:", len(shacl_accumulator))
    return shacl_accumulator

if __name__ == "__main__":
    from rag import split_pdf, summarize, embedding

    # Load and summarize the PDF document
    file = "./content/rinf_application_guide_for_register_en_0_test.pdf"
    texts, tables, table_chunks, images = split_pdf(file)
    text_summaries, table_summaries, image_summaries = summarize(texts, tables, table_chunks, images)

    # Embed the content
    retriever = embedding(texts, text_summaries, tables, table_summaries, images, image_summaries)

    # Run the graph
    shacl_constraints = run_shacl_generation(retriever)

    print(shacl_constraints)
