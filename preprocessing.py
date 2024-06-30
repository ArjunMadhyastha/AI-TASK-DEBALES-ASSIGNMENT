from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle


print("URL Loading...")
urls =["https://brainlox.com/courses/category/technical"]
loader = SeleniumURLLoader(urls=urls)
data = loader.load()

print("Text is being splitted")
text_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n\n','\n','.',','],
    chunk_size = 400
)
docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(docs,embeddings)
file_path = "storing_vector_store.pkl"

with open(file_path,"wb") as f:
    pickle.dump(vector_store,f)