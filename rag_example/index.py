from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

DB_PATH = "./chroma_db"

# loader = WebBaseLoader(
#     web_paths=("https://www.fernuni-hagen.de/english/research/clusters/index.shtml",)
# )
# docs = loader.load()


urls = ["https://www.fernuni-hagen.de/english/research/clusters/index.shtml", "https://www.fernuni-hagen.de/english/research/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

print(len(docs_transformed[0].page_content))
print(docs_transformed[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs_transformed)

vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model='mistral'), persist_directory=DB_PATH)