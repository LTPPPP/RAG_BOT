import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import Client
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define file paths and settings
dataset_path = "dataset.txt"
index_name = "infinity-demo"
persist_directory = "./chroma_data"

# Ensure persist directory exists
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Load and split documents
loader = TextLoader(dataset_path, encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Initialize embeddings and Chroma
embeddings = HuggingFaceEmbeddings()
chroma_client = Client(Settings(persist_directory=persist_directory))

# Check if collection exists and create/load accordingly
existing_collections = [collection.name for collection in chroma_client.list_collections()]
if index_name not in existing_collections:
    docsearch = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=index_name, persist_directory=persist_directory)
else:
    docsearch = Chroma(collection_name=index_name, persist_directory=persist_directory)

docsearch.persist()
print(f"Index '{index_name}' created and persisted successfully.")

# Initialize LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_k": 50, "top_p": 0.95, "max_length": 1000},
    huggingfacehub_api_token= api_token
)

# Define prompt template
template = """
You are a fortune teller. These Human will ask you questions about their life.
Use the following piece of context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer within 3 sentences and concise.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Define ChatBot class
class ChatBot:
    def __init__(self, docsearch, llm, prompt):
        self.rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke({"context": "", "question": question})

# Create chatbot instance and ask a question
bot = ChatBot(docsearch, llm, prompt)
question = "Nguyên nhân bệnh tự kỷ là gì?"
result = bot.ask(question)

print(result)
