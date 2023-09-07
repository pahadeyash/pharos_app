from flask import Flask, request, send_from_directory
from flask_cors import CORS
import os
import json
import pdfplumber
import requests
import openai
import boto3 
from uuid import uuid4
from io import BytesIO, StringIO
import datetime
import shutil
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

import os

access_key = os.environ.get('ACCESS_KEY')
secret_key = os.environ.get('SECRET_KEY')
region = os.environ.get('REGION')
openai_api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__, static_folder="frontend/build", template_folder="frontend/build")
CORS(app)
s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region
)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

@app.route('/<user_id>/upload', methods=['POST'])
def upload(user_id):
    if 'file' not in request.files:
        return 'No file found', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    try:
        s3_client.upload_fileobj(
            file,
            "test-scripts-uploads",
            f'{user_id}/{file.filename}'
        )
    except Exception as e:
        print("Something Happened: ", e)
        return str(e)

    # Create a temporary file path
    temp_path = f'/tmp/{uuid4().hex}_{file.filename}'
    
    try:
        # Download the PDF file from S3 and save it to the temporary path
        s3_client.download_file(
            "test-scripts-uploads",
            f'{user_id}/{file.filename}',
            temp_path
        )
        
        # Process the PDF file with PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        # Combine the pages, and replace the tabs with spaces
        text = ""

        for page in pages:
            text += page.page_content

        text = text.replace('\t', ' ')

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
        docs = text_splitter.create_documents([text])
        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents([x.page_content for x in docs])

        num_clusters = 7
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

        # Find the closest embeddings to the centroids

        # Create an empty list that will hold your closest points
        closest_indices = []

        # Loop through the number of clusters you have
        for i in range(num_clusters):

            # Get the list of distances from that particular cluster center
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

            # Find the list position of the closest one (using argmin to find the smallest distance)
            closest_index = np.argmin(distances)

            # Append that position to your closest indices list
            closest_indices.append(closest_index)
        
        selected_indices = sorted(closest_indices)

        llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=500,
                 model='gpt-4'
                )

        map_prompt = """
        You will be given a single passage of a film script. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.

        ```{text}```
        FULL SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_chain = load_summarize_chain(llm=llm3, chain_type="stuff", prompt=map_prompt_template)
        selected_docs = [docs[doc] for doc in selected_indices]

        # Make an empty list to hold your summaries
        summary_list = []

        # Loop through a range of the lenght of your selected docs
        for i, doc in enumerate(selected_docs):
            # Go get a summary of the chunk
            chunk_summary = map_chain.run([doc])

            # Append that summary to your list
            summary_list.append(chunk_summary)

        summaries = "\n".join(summary_list)

        # Convert it back to a document
        summaries = Document(page_content=summaries)
        
        llm4 = ChatOpenAI(temperature=0,
                 openai_api_key="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",
                 max_tokens=3000,
                 model='gpt-4',
                 request_timeout=120
                )

        combine_prompt = """
        You will be given a series of summaries from a script. The summaries will be enclosed in triple backticks (```)
        Acting as a script coverage consultant, could you please provide script coverage of the script?
        In particular, include a detailed summary of the series of summaries of the script using the 3-act structure commonly associated with filmmaking.

        ```{text}```
        VERBOSE SUMMARY:
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

        reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template)
        summary = reduce_chain.run([summaries])

    except Exception as e:
        print("Something Happened: ", e)
        return str(e)
    
    finally:
        # Delete the temporary file
        os.remove(temp_path)
    
    return {
        "user_id": user_id,
        "script":  file.filename,
        "summary": summary
    }

@app.route('/', defaults={'path': ''})

@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
if __name__ == '__main__':
    app.run()