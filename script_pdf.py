
import PyPDF2
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Mise à jour de l'importation
from rag_system import RAG
rag= RAG(

)
embeddings_model = "Sahajtomar/french_semantic"
embedder = HuggingFaceEmbeddings(model_name=embeddings_model, cache_folder="./cache_folder")

pdf_path = "brochure_cancer.pdf"

def pdf_to_docs(pdf_path: str) -> List[Document]:
    """Converts a PDF file to a list of Documents with metadata."""
    doc_chunks = []

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        # Iterate through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=20,
            )
            chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                )
                doc_chunks.append(doc)

    return doc_chunks

try:
    result = pdf_to_docs(pdf_path)
    print(f"Successfully processed {len(result)} chunks from the PDF.")
except Exception as e:
    print(f"Error processing PDF: {e}")


# Stocker les chunks dans la base de données
def store_chunk(pdf_path):
    data = pdf_to_docs(pdf_path)
    print("Saving embeddings in progress !!!")
    # La persistance sera automatique avec l'argument persist_directory
    db = Chroma.from_documents(documents=data, embedding=rag.embedder, persist_directory=rag.persist_directory)
    print("Saving embedding_model in progress !!!")
    return db

# Appeler la fonction pour stocker les chunks
vectorstore = store_chunk(pdf_path)


# db = Chroma(rag.persist_directory, embedding_function=rag.embedder)
query = "Qu'est ce que le cancer?"
matched_docs = vectorstore.similarity_search(query, k=5)

# Vérifier si des documents correspondants ont été trouvés
if matched_docs:
    context = "\n".join([doc.page_content for doc in matched_docs])
    print(context)
else:
    print("Aucun document correspondant n'a été trouvé.")