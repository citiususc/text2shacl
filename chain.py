from rdflib import Graph, RDF, RDFS, OWL
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from base64 import b64decode
from rag import split_pdf, summarize, embedding


# Function to process the documents retrieved from the RAG
def parse_docs(docs):
    """Classify documents into images and texts"""
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)  # Try to decode as base64
            b64.append(doc)  # If valid, it is an image
        except Exception:
            text.append(doc)  # If it fails, it is a text
    return {"images": b64, "texts": text}

# Function to build the SHACL shape generation prompt
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    entity = kwargs["entity"]

    # Initialize the context_text variable to concatenate all text content
    context_text = ""
    if docs_by_type["texts"]:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text + "\n\n"  # Only append text (string) from documents

    # Build the SHACL prompt
    prompt_template = f"""
    You are an expert in semantic web technologies and constraints.

    Based on the following context, identify and describe the data restrictions for the entity "{entity['name']}".

    Entity Description: {entity['description']}

    Context:
    {context_text}

    Please return the data restrictions in a **tabular format**, with the following columns:
    - Restriction Name
    - Description

    **Only return the table below. Do not include any explanations, comments, or additional information.**

    Example format:
    | Restriction Name | Description |
    |------------------|------------|
    """


    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_template)
    ])


if __name__ == "__main__":
    # Load the ontology from a TTL file
    g = Graph()
    g.parse("https://data-interop.era.europa.eu/era-vocabulary/ontology.ttl", format="turtle")

    # Query all classes in the ontology
    query_classes = """
        SELECT ?class ?comment WHERE {
            ?class a owl:Class .
            FILTER(!isBlank(?class))  # Excluye blank nodes
            OPTIONAL { ?class rdfs:comment ?comment . }
            FILTER (lang(?comment) = "en" || !BOUND(?comment))  # Solo comentarios en inglés o sin comentario
        }
    """

    class_info = []
    for row in g.query(query_classes):
        class_uri = row[0]
        comment = row[1] if row[1] else "No description available"
        class_info.append({"name": str(class_uri), "description": str(comment)})

    # Process the PDF document to extract text, tables, and images
    file = "./content/rinf_application_guide_for_register_en_0_test.pdf"
    texts, tables, table_chunks, images = split_pdf(file)
    text_summaries, table_summaries, image_summaries = summarize(texts, tables, table_chunks, images)

    # Generate embeddings for the text, tables, and images
    retriever = embedding(texts, text_summaries, tables, table_summaries, images, image_summaries)

    # Define the processing chain (RAG + SHACL)
    chain = (
        RunnableLambda(lambda entity: {
            "context": parse_docs(retriever.invoke(entity['name'])),
            "entity": entity
        })
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    # Run SHACL shape generation for each entity
    for entity in class_info:
        print(f"\n  Generating SHACL for: {entity['name']}")

        # Generate SHACL shape using the language model
        shacl_shape = chain.invoke(entity)

        print(shacl_shape)
        print("\n" + "-" * 80)