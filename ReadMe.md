BackendRAG is an advanced backend server application that leverages the power of AI through various libraries including LangChain, Sentence Transformers, and Flask to automate document processing and validation tasks. It integrates multiple functionalities for document extraction, text splitting, embedding, and structured data storage using Chroma.

**Features**

* Document Processing: Supports various formats like PDF, DOCX and txt files

* Text Extraction and Splitting: Utilizes custom text splitters for handling large texts.

* Embeddings and Validation: Employs Sentence Transformers for generating embeddings and cosine similarity for document validation.

* Creation of a vector database using CHROMA DB for storing the structures of documents extracted from reference files submitted by users

* Flask API: A simple API to handle document validation requests dynamically.

* Dynamic Category Structuring: Automatically processes reference documents to create and store structured data

* Session based memory management: Manages document validation results and user interactions in memory on a per-session basis, enabling seamless tracking of session- specific data.

**Installation and Test**

**Steps:**

* pip install \-r requirements.txt

* python backendrag.py



* streamlit run frontendrag.py



* Ensure the environment variables are set as required:

\`GROǪ\_API\_KEY\` for LangChain integration.

\`HF\_API\_KEY\` for using Hugging Face models.


**API Architecture**



* File Upload: The process begins when the user uploads a file to be validated through the user interface.

* API Request: A request is made to the API. This request sends the file path and the specified category to the API.

* Retrieval of Category Data: The API queries the vector database to extract the appropriate category structure associated with the file's characteristics.

* Validation Report Generation: With the determined category, an enriched request is sent to the Llama model to generate a validation report.

* Presentation and Storage of Results\*: The validation report is then presented to the user via a Streamlit interface

