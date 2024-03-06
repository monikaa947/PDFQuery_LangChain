import os
# LangChain components to use
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores.cassandra import Cassandra
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Support for dataset retrieval with Hugging Face
from datasets import load_dataset
from dotenv import load_dotenv

# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:

# read text inside the pdf 
from PyPDF2 import PdfReader

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# provide the path of  pdf file/files.
pdfreader = PdfReader('sample_data/sample_contract.pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Initialize the connection to your database:
# cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects for later usage:
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name="langchain",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)
print(f"Connected to Astra DB: {vstore.get_collections()}")
# Create your LangChain vector store ... backed by Astra DB!
# astra_vector_store = Cassandra(
#     embedding=embedding,
#     table_name="qa_mini_demo",
#     session=None,
#     keyspace=None,
# )

from langchain.text_splitter import CharacterTextSplitter
# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

inserted_ids = vstore.add_documents(texts)
print(f"\nInserted {len(inserted_ids)} documents.")
# Load the dataset into the vector store
# astra_vector_store.add_texts(texts[:50])
# print("Inserted %i headlines." % len(texts[:50]))
# astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


