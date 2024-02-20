from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "./chroma_db"

path = "/Users/zesch/dkpro/Studienbriefe/deWitt/25104_LE1_Bildung in der digitalisierten Gesellschaft_Final_SoSe23.pdf"
loader = PyPDFLoader(path)
docs = loader.load_and_split()

print(len(docs[0].page_content))
print(docs[0].page_content[:500])

vectorstore = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model='mistral'), persist_directory=DB_PATH)