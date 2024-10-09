import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# Define FAISS vector store path
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Define custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Define Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load the LLM model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load FAISS vector store
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Load the LLM model
    llm = load_llm()

    # Set the custom prompt
    qa_prompt = set_custom_prompt()

    # Create the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Function to handle query and return result
def final_result(query):
    qa = qa_bot()
    response = qa({'query': query})
    return response['result'], response.get('source_documents', [])

# Streamlit App Interface
def main():
    st.title("Q/A About Alzheimer's Disease:")

    # Text input for query
    user_query = st.text_input(" WHAT DO YOU  WANT TO KNOW :")

    if st.button("Submit"):
        if user_query:
            st.write("Processing your query...")
            result, sources = final_result(user_query)
            
            # Display result
            st.subheader("Answer:")
            st.write(result)

            # Display sources
            if sources:
                st.subheader("Source Documents:")
                for source in sources:
                    st.write(f"- {source.metadata.get('source', 'Unknown source')}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

