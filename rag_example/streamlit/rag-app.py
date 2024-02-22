# adapted from https://github.com/aigeek0x0/rag-with-langchain-colbert-and-ragatouille
#!/bin/python3.11
import os
import tempfile
import streamlit as st
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from ragatouille import RAGPretrainedModel

favicon = Image.open("fuh.png")

st.set_page_config(page_title="RAG with Mistral 7B and ColBERT", page_icon=favicon)
st.sidebar.image("fuh.png", use_column_width=True)
with st.sidebar:
    st.write("**RAG with Mistral 7B and ColBERT**")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        print("here you are")
        loader = PyPDFLoader(temp_filepath)
        print("done")
        docs.extend(loader.load())

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectordb = Chroma.from_documents(splits, embeddings)

    # define retriever
    chroma_retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
    )

    ## ragatouille/colbert implmentation
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/langchain-index")
    ragatouille_retriever = RAG.as_langchain_retriever(k=10)

    ### initialize the ensemble retriever
    retriever = EnsembleRetriever(retrievers=[chroma_retriever, ragatouille_retriever],
                                            weights=[0.50, 0.50])
    return retriever


uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)
st.write("uploaddone:-")
llm = Ollama(model="mistral")

msgs = StreamlitChatMessageHistory()

## prompt template
RESPONSE_TEMPLATE = """[INST]
<>
You are a helpful AI assistant.

Use the following pieces of context to answer the user's question.<>

Anything between the following `context` html blocks is retrieved from a knowledge base.


    {context}


REMEMBER:
- If you don't know the answer, just say that you don't know, don't try to make up an answer.
- Let's take a deep breath and think step-by-step.

Question: {question}[/INST]
Helpful Answer:
"""

PROMPT = PromptTemplate.from_template(RESPONSE_TEMPLATE)
PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT,
    }
)

if len(msgs.messages) == 0 or st.sidebar.button("New Chat"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        response = qa_chain({"query": user_query})

        ## print answer
        answer = response["result"]
        st.write(answer)

about = st.sidebar.expander("About")
about.write("You can easily chat with a PDF using this AI chatbot.")
