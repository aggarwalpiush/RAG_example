## ragatouille/colbert implmentation
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import PyPDFLoader

path = "/Users/zesch/dkpro/Studienbriefe/deWitt/25104_LE1_Bildung in der digitalisierten Gesellschaft_Final_SoSe23.pdf"
loader = PyPDFLoader(path)
r_docs = loader.load_and_split()

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

ragatouille_docs = [str(doc) for doc in r_docs]

RAG.index(
  collection=ragatouille_docs,
  index_name="langchain-index",
  max_document_length=512,
  split_documents=True,
)