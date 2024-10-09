Alzheimer's Disease Expert Chatbot Project Documentation
Overview
This project aims to create an expert chatbot specializing in answering questions related to Alzheimer's disease. The chatbot is built using a Retrieval-Augmented Generation (RAG) model, which enhances the chatbot's responses by retrieving relevant information from a collection of PDFs related to Alzheimer's disease. Two implementations have been developed:

Model.py: Uses Chainlit for the chatbot interface.
Model1.py: Uses Streamlit for a user-friendly web interface.
Both implementations utilize a FAISS vector store to efficiently retrieve the most relevant documents for answering user queries, ensuring that the responses are both accurate and supported by reliable sources.

Key Features
Document Loader: The project combines several Alzheimer's-related PDF documents to build the knowledge base for the RAG agent.

Vector Store: FAISS (Facebook AI Similarity Search) is used to create a vectorized representation of the documents for fast retrieval of relevant information.

Custom Prompt Template: A custom prompt ensures that the chatbot only provides helpful and accurate responses. If the chatbot doesn't know an answer, it is programmed to acknowledge that.

Model Loading:

The project uses the HuggingFace Embeddings model for vectorizing the document content.
The LLM (Large Language Model) used is Llama-2-7b loaded through CTransformers, which powers the natural language understanding and response generation.
Chain Construction: A retrieval-based question-answering chain is created using Langchain, enabling the chatbot to fetch the most relevant information from the knowledge base in response to user queries.

User Interface:

Streamlit provides a web interface allowing users to input questions and receive detailed responses, along with the source documents for transparency.
Chainlit is used in a parallel model.py file for those who prefer a more streamlined command-line interaction with the chatbot.