
# Routes for embedding and querying
import os
import urllib.parse
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma

from src.constants import CHROMA_SETTINGS

app = FastAPI()


load_dotenv()

EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME")
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY")

MODEL_TYPE = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")


async def test_embedding():
    # Create the folder if it doesn't exist
    os.makedirs(source_directory, exist_ok=True)
    # Create a sample.txt file in the source_documents directory
    file_path = os.path.join("source_documents", "test.txt")
    with open(file_path, "w") as file:
        file.write("This is a test.")
    # Run the ingest.py command
    os.system("python src/ingest.py --collection test")
    # Delete the sample.txt file
    os.remove(file_path)
    print("embeddings working")


async def model_download(model_type: str = MODEL_TYPE):
    match model_type:
        case "LlamaCpp":
            url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
        case "GPT4All":
            url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    folder = "models"
    filename = os.path.join(folder, url.split("/")[-1])
    # Check if the file already exists
    if os.path.exists(filename):
        print("File already exists.")
        return
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    # Run wget command to download the file
    os.system(f"wget {url} -O {filename}")
    # Set the MODEL_PATH environment variable
    os.environ["MODEL_PATH"] = filename
    print("model downloaded")


# Starting the app with embedding and llm download
@app.on_event("startup")
async def startup_event():
    await test_embedding()
    await model_download()


# Example route
@app.get("/")
async def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}


@app.post("/embed")
async def embed(files: List[UploadFile], collection_name: Optional[str] = None):
    saved_files = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        if collection_name is None:
            # Handle the case when the collection_name is not defined
            collection_name = file.filename
    print("Files saved successfully")
    os.system(f"python src/ingest.py --collection {collection_name}")

    # Delete the contents of the folder
    for file in files:
        os.remove(os.path.join(source_directory, file.filename))

    return {"message": "Files embedded successfully", "saved_files": saved_files}


@app.post("/retrieve")
async def query(query: str, collection_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    match MODEL_TYPE:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, backend="gptj", callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {MODEL_TYPE} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Get the answer from the chain
    res = qa(query)
    print(res)
    answer, docs = res["result"], res["source_documents"]

    return {"results": answer, "docs": docs}
