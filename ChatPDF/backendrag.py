import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import pandas as pd



#Variables globales


store = {}

history=""
splitter = RecursiveCharacterTextSplitter( chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False)


embedding_function = HuggingFaceInferenceAPIEmbeddings(
    api_key='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',
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

db = Chroma(
    persist_directory="chroma",  

    embedding_function=embedding_function
)

categories = []  # List to store category names
structures_text = []


reference_folder = os.path.join(os.getcwd(), 'references_file')
os.makedirs(reference_folder, exist_ok=True)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
os.environ['GROQ_API_KEY'] = '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'
groq_api_key = os.environ['GROQ_API_KEY']
chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

'''Fonctions pour le traitement des fichiers de references'''

def split_text_data(text_data):
    try:
        text_chunks = splitter.split_text(text=text_data)
        return text_chunks
    except Exception as e:
        print(f"Error splitting text data: {e}")
        return []





def load_documents_into_chroma(db, documents):
    try:
        uuids = [str(uuid4()) for _ in documents]  # Generating unique IDs for each document
        db.add_documents(documents=documents, ids=uuids)
        print(f"Successfully added  documents to Chroma.")
    except Exception as e:
        print(f"Failed to add documents to Chroma: {e}")





def extract_text(file_path):
    file_type = file_path.split('.')[-1].lower()
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'docx':
        loader = Docx2txtLoader(file_path)
    elif file_type == 'txt':
        loader = TextLoader(file_path)
    else:
        return None
    
    documents = loader.load_and_split() if hasattr(loader, 'load_and_split') else loader.load()
    return " ".join(doc.page_content for doc in documents)

def extract_structure(document_text, category):
    prompt_template = ChatPromptTemplate.from_template("""
      
Imagine you are creating a template for a document categorized under '{category}'. Analyze the example provided and describe the typical structure of such a document based on the example text: '{context}'. Your task is to outline the typical fields found in this document type, such as name, date of birth, and personal image. Ensure your description specifies the fields generally included in this category of documents but avoid using specific data from the example text.
    """)
    # Use the custom JsonOutputParser in the chain
    chain = LLMChain(llm=chat, prompt=prompt_template)
    structured_info = chain.run({"context": document_text, "category": category})
    return structured_info
def process_reference_files(reference_folder):
    reference_files = [
        f for f in os.listdir(reference_folder) 
        if not f.endswith('structure') and not f.endswith('.txt')
    ]
    category_info = {}
    
    for ref_file in reference_files:
        category = ref_file.split('.')[0]  # Assuming file name indicates category
        file_path = os.path.join(reference_folder, ref_file)
        document_text = extract_text(file_path)
        
        if document_text:
            structured_info = extract_structure(document_text, category)
            category_info[category] = structured_info
    
    return category_info

def create_structures_files_if_not_exist(reference_folder):
    os.makedirs(reference_folder, exist_ok=True)  # Ensure the directory exists

    category_info = process_reference_files(reference_folder)
    for category, info in category_info.items():
        text_file_path = os.path.join(reference_folder, f'{category}_structure.txt')
        if not os.path.exists(text_file_path):
            with open(text_file_path, 'w') as text_file:
                text_file.write(info)  # Write the string directly to the file




    



'''Fonctions pour la gestion de la memoire '''
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def add_validation_result(session_id, validation_report, is_valid, comments=None):
    history = get_by_session_id(session_id)
    validation_message = f"********************************************** Example of Validation Report: {validation_report} *********************************************"
    history.add_message(AIMessage(content=validation_message))




def get_recent_validations(session_id):
    history = get_by_session_id(session_id)
    return "\n".join([msg.content for msg in history.messages])  # Utilise directement l'attribut messages




'''Fonctions pour l initialisation de l api '''


def load_all_category_structures():
    documents = []
    reference_folder = os.path.join(os.getcwd(), 'references_file')
    for filename in os.listdir(reference_folder):
        if filename.endswith('_structure.txt'):
            category_name = filename.replace('_structure.txt', '')
            file_path = os.path.join(reference_folder, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                document = Document(
                    page_content=content,
                    metadata={"category": category_name},
                    id=str(uuid4())
                )
                documents.append(document)
            if documents:
                print('documents crees')
            if  documents and not os.listdir(os.path.join(os.getcwd(), 'chroma')):
                load_documents_into_chroma(db, documents)
            else:
                print("No documents were created from the reference files.")
            categories.append(category_name)
            

def initialize_model_context():
   
    context = ""
    relevance_threshold = 0.5
    for category in categories:  # Utilisez la liste de catégories existante
        query_text = f"Structure pour la catégorie {category}"
        results = db.similarity_search_with_relevance_scores(query_text, k=5)  # Récupère les 5 résultats les plus pertinents
        
        # Filtre les résultats basés sur le seuil de pertinence
        filtered_results = [(doc, score) for doc, score in results if score > relevance_threshold]
        
        if not filtered_results:
            context += f"\nCategory: {category}\nStructure:\nNot found\n"
        else:
            # Concatène les résultats filtrés en un seul texte de contexte
            structure_texts = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
            context += f"\nCategory: {category}\nStructure:\n{structure_texts}\n"
    prompt_template = ChatPromptTemplate.from_template("""
       You are a document validation assistant. Your task is to analyze and validate documents 
    based on predefined structures for different categories. Here are the structures you need to know 
    for each category, and how you should use this information: '{context}'
        Based on this example, please format your response as a JSON schema describing what fields 
        (like name, date of birth, personal image, etc.) are generally included, without providing 
        specific data values from this text.
                                                        
        When validating a document:
    1. Check that the document contains all the sections and information required as described for its category.
    2. Identify any non-compliance or omissions in the structure or content.
    3. Write a validation report indicating whether the document is valid or not, with details on any points of non-compliance.

    This role requires meticulous attention to detail and a precise understanding of the expectations for each document category.
                                                        

    """)

    chain = LLMChain(llm=chat, prompt=prompt_template)
    structured_info = chain.run({"context": context} )
    return structured_info

'''Fonctions de validation'''





def validate_document(document_text, category):
    try:
        prompt_template = ChatPromptTemplate.from_template(f"""
            En utilisant la structure préalablement chargée pour la catégorie '{category}', veuillez vérifier les points suivants :
            1. Le document ci-dessous appartient-il à la catégorie '{category}' ?
            2. Respecte-t-il la structure prédéfinie de cette catégorie ?

            Document à valider :
            {document_text}

            Fournissez un rapport de validation détaillé indiquant si le document est valide ou non c est a dire il est de la meme categorie mentionnee, vous trouverez ici l history de vos reponses precedentes '{history}'
        """)
        chain = LLMChain(llm=chat, prompt=prompt_template)
        validation_report = chain.run({"category": category, "document_text": document_text,"history":history})
        return validation_report
    except Exception as e:
        print(f"An error occurred during document validation: {e}")
        return {"error": "Failed to validate document due to internal error."}

def validate_document_concisely(document_text, category):
    detailed_report = validate_document(document_text, category)
    
    if isinstance(detailed_report, dict) and "error" in detailed_report:
        return detailed_report

    prompt_template = ChatPromptTemplate.from_template("""
        Given the detailed validation report below for the document category '{category}', provide a single word indicating if the document is "Valid" or "Not Valid".

        Detailed Validation Report:
        {detailed_report}
    """)
    chain = LLMChain(llm=chat, prompt=prompt_template)
    concise_result = chain.run({"category": category, "detailed_report": detailed_report}).strip()
    return concise_result



@app.route('/update-category', methods=['POST'])
def update_category():
    
    data = request.get_json()
    new_category_name = data['category_name']
    reference_folder = os.path.join(os.getcwd(), 'references_file')
    json_file_path = os.path.join(reference_folder, f'{new_category_name}_structure.txt')
    
    try:
        if not os.path.exists(json_file_path):
            pdf_file_path = os.path.join(reference_folder, f'{new_category_name}.pdf')
            document_text = extract_text(pdf_file_path)
            structured_info = extract_structure(document_text, new_category_name)
            
            with open(json_file_path, 'w') as text_file:
                text_file.write(structured_info) 
            
            # Ajouter les informations structurées à Chroma
            document = {
                "category": new_category_name,
                "content": structured_info,
                "id": str(uuid4())
            }
            load_documents_into_chroma(db, [document])
            initialize_model_context()
            return jsonify({"status": "success", "message": "Category updated and added to Chroma successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "Category already exists"}), 409
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_documents():
    data = request.get_json()
    session_id = data.get('session_id', 'default_session')
    document_path = data['uploaded_file_path']
    document_category = data['document_category']
    history = get_by_session_id(session_id)
    document_text = extract_text(document_path)
    
    if document_text:
        detailed_report = validate_document(document_text, document_category)
        concise_result = validate_document_concisely(document_text, document_category)
        return jsonify({
            "validation_report": detailed_report,
            "concise_result": concise_result
        })
    else:
        return jsonify({"error": "Invalid document format"}), 400

if __name__ == '__main__':
  

    create_structures_files_if_not_exist(reference_folder)
    load_all_category_structures()
    initialize_model_context()
    app.run(debug=True)






























