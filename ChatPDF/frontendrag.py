import streamlit as st
import requests
import os

# Chemin pour stocker les fichiers de référence
REFERENCES_FOLDER = "references_file"
if not os.path.exists(REFERENCES_FOLDER):
    os.makedirs(REFERENCES_FOLDER)

UPLOAD_FOLDER = "/tmp/document_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.title("Document Validator")

def send_category_update(category_name):
    url = 'http://localhost:5000/update-category'  # Assurez-vous que l'URL est correcte
    data = {'category_name': category_name}
    response = requests.post(url, json=data)
    return response.status_code

with st.sidebar:
    st.header("Add New Document Category")
    new_category_name = st.text_input("Enter the new category name")
    reference_file = st.file_uploader("Upload reference file for the new category", type=['pdf'], key="refUploader")
    
    if st.button("Add Category"):
        if new_category_name and reference_file:
            file_path = os.path.join(REFERENCES_FOLDER, f"{new_category_name}.pdf")
            with open(file_path, "wb") as f:
                f.write(reference_file.getbuffer())
            # Envoi de la notification au backend après l'ajout du fichier
            response_code = send_category_update(new_category_name)
            if response_code == 200:
                st.success(f"New category '{new_category_name}' added and backend updated.")
            else:
                st.error("Please refresh page  to update backend. .")
        else:
            st.error("Please enter a category name and upload a reference file.")

# Formulaire principal pour la validation de documents
categories = os.listdir(REFERENCES_FOLDER)
categories = [cat.replace(".pdf", "") for cat in categories if cat.endswith(".pdf")]

document_category = st.selectbox('Select the category of your file', categories)
uploaded_file = st.file_uploader('Upload the file to validate', type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'])

def save_file_locally(file):
    if file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    return None

if st.button("Validate Document"):
    if uploaded_file:
        document_path = save_file_locally(uploaded_file)
        data = {
            "uploaded_file_path": document_path,
            "document_category": document_category
        }
        response = requests.post("http://localhost:5000/process", json=data)
        if response.status_code == 200:
            response_data = response.json()
            st.write("Validation Report:", response_data['validation_report'])
            concise_result = response_data.get('concise_result', '').strip()
            st.write(f"Verification: {concise_result}")

            if "valid" in concise_result.lower() and "not valid" not in concise_result.lower():
                st.success("The uploaded document is valid.")
            elif "not valid" in concise_result.lower() or "invalid" in concise_result.lower():
                st.error("The uploaded document is not valid.")
            else:
                st.error("Unexpected response format.")
        else:
            st.error("Failed to process document due to: " + response.text)
    else:
        st.error("Please upload a document.")
