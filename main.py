from rag import split_pdf, summarize, embedding

# For printing images
import base64
import io
from PIL import Image


if __name__ == "__main__":
    file = "./content/rinf_application_guide_for_register_en_0_test.pdf"
    texts, tables, table_chunks, images = split_pdf(file)
    text_summaries, table_summaries, image_summaries = summarize(texts, tables, table_chunks, images)

    retriever = embedding(texts, text_summaries, tables, table_summaries, images, image_summaries)

    docs = retriever.invoke("What's an Operational Point? What types of OP exist?")

    for doc in docs:
        try:
            # Try to decode as base64
            decoded_bytes = base64.b64decode(doc)
            
            # Try to open as an image
            image = Image.open(io.BytesIO(decoded_bytes))
            image.show()
        except Exception:
            # If it fails, print as normal text
            print(doc)
        
        print("\n\n" + "-" * 80)