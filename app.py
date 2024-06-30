from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import pickle
import os
from flask import Flask,request,jsonify
from flask_restful import Api, Resource
import requests
import json 

app = Flask(__name__)
api = Api(app)

file_path = "storing_vector_store.pkl"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

def chain_response(query):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(),return_source_documents=True)
            result = chain({"query": query}, return_only_outputs=True)
            return result["result"]
query = input("Enter the query: ")
@app.route('/chat',methods=['POST'])
def chat():
    response = chain_response(query)
    return jsonify({"response": response})

if __name__=='__main__':
    app.run(debug=True)