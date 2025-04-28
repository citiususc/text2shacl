# For splitting
from unstructured.partition.pdf import partition_pdf

# For summarizing
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# For embedding
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
def split_pdf(file_path: str):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    )

    # separate tables from texts
    tables = []
    table_chunks = []
    texts = []
    images_b64 = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)


        if "CompositeElement" in str(type((chunk))):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
            texts.append(chunk)

        if "TableChunk" in str(type((chunk))):
            table_chunks.append(chunk)

    return texts, tables, table_chunks, images_b64


def summarize(texts, tables, table_chunks, images):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """

    prompt_table = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """

    prompt_table_chunk = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """

    prompt_image = """Describe the image in detail. 
    For context, the images are part of a document from the European Union Agency for Railways, specifically of RINF (Registers of Infrastructure). 
    If the image is a plot, be specific about graphs, such as bar plots."""


    # Set the model
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")

    # Summarize texts
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    # Summarize images
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_image},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    image_summaries = chain.batch(images)

    return text_summaries, table_summaries, image_summaries


def embedding(texts, text_summaries, tables, table_summaries, images, image_summaries):
    # Chroma will store vectors and support vector-based similarity search.
    vectorstore = Chroma(collection_name="text2shacl", embedding_function=OpenAIEmbeddings())

    # Documents are stored temporarily in memory, not persistently.
    store = InMemoryStore()

    # The key used to uniquely identify each document in the docstore.
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    if len(text_summaries)>0:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    if len(table_summaries)>0:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    if len(image_summaries)>0:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))

    return retriever

