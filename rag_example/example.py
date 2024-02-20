import bs4
from langchain import hub
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import Ollama
from index import DB_PATH

vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model='mistral'))
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt-mistral")
llm = Ollama(model="mistral")

query = "Was sagt Rousseau im Gesellschaftsvertrag?"
retrieved_docs = retriever.get_relevant_documents(
    query
)
print(len(retrieved_docs))
print(retrieved_docs[0].page_content[:500])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke(query))