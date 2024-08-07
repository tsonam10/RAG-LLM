from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from ingestion import get_embedding_function, load_documents, add_to_chroma
import json 
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from ingestion import create_llm
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st 

# set page config 
st.set_page_config(page_title="Document Q&A", page_icon="📚")

# Initalize session state 
if 'initialized' not in st.session_state:
    st.session_state.initialzed =False 
    st.session_state.llm =None 
    st.session_state.retriever=None 


# Initialzed function 
@st.cache_resource
def initialize_app():
    embeddings = get_embedding_function()
    texts = load_documents()
    vector_store = add_to_chroma(texts, embeddings)
    llm = create_llm()
    load_vector_store =Chroma(persist_directory='store/pet_cosine', embedding_function=embeddings)
    retriever  =load_vector_store.as_retriever(search_kwargs={"k":1})
    return llm, retriever

# app logic 

def main():
    st.title("Document Q&A System")

    if not st.session_state.initialized:
        with st.spinner("Initializing the application...."):
            st.session_state.llm, st.session_state.retriever=initialize_app()
            st.session_state.initialized=True 

    user_question = st.text_input("Enter your question")

    if user_question:
        with st.spinner("Searching for an answer...."):
            prompt_template = """
            Use the following pieces of information to answer the user's questions

            Context: {context}
            Question: {question}

            Only return the helpful answer below and nothing else. 
            Helpful answer:
            """

            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

            chain_type_kwargs = {'prompt': prompt}
            qa = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff", 
                retriever=st.session_state.retriever, 
                return_source_documents=True, 
                chain_type_kwargs=chain_type_kwargs, 
                verbose=True
            )
            
            response = qa(user_question)
            
            st.subheader("Answer:")
            st.write(response['result'])
            
            st.subheader("Source Document:")
            st.write(response['source_documents'][0].page_content)
            
            st.subheader("Document Source:")
            st.write(response['source_documents'][0].metadata['source'])


if __name__=="__main__":
    main()
