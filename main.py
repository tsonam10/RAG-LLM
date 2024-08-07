from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from ingestion import get_embedding_function
from ingestion import load_documents
from ingestion import add_to_chroma
import json 
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from ingestion import create_llm
from flask import Flask, request, jsonify, render_template
#from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings



app = Flask(__name__)

prompt_template ="""Use the following pieces of information to answer the user's questions

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. 
Helpful answer:
"""

embeddings = get_embedding_function()
texts = load_documents()
vector_store = add_to_chroma(texts, embeddings)
llm = create_llm()


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

load_vector_store =Chroma(persist_directory='store/pet_cosine', embedding_function=embeddings)

retriever = load_vector_store.as_retriever(search_kwarg={"k":1})

@app.route('/')
def index():
    return render_template('template/index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')
    chain_type_kwargs ={'prompt': prompt}
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     return_source_documents=True, 
                                     chain_type_kwargs=chain_type_kwargs, 
                                     verbose=True)
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    return jsonify(response_data)

if __name__=="__main__":
    app.run()
