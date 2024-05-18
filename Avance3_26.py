import os
# import chromadb
import pandas as pd

# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.text_splitter import CharacterTextSplitter

# from langchain.document_loaders import DataFrameLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    This function reads a CSV file from the specified file path and 
    returns the data as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def load_csv_with_loader():
    """
    Loads data from a CSV file using a CSVLoader.

    Args:
        filepath (str): The path to the CSV file to be loaded.

    Returns:
        Any: The data loaded from the CSV file.
    """
    loader = CSVLoader(file_path="docs.csv", source_column="page_content", encoding="utf8")
    data = loader.load()

    return data


def create_documents(data, data_columns):
    data["page_content"] = ""
    for column in data_columns:
        data["page_content"] += f"{column}: " + data[column].astype(str) + "\n"

    docs = data[["page_content"]]

    # docs = DataFrameLoader(
    #     data,
    #     page_content_column="page_content"
    # ).load()

    # ids = data.index.astype(str).to_list()

    # return ids, docs

    docs.to_csv("docs.csv")

    return docs


def get_retriever(docs: pd.DataFrame):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "data/embeddings_vector_store")
    
    embedding_model = FastEmbedEmbeddings()
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embedding_model,
        persist_directory = data_path,
    )

    retriever = vector_store.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {
            "k": 20,
            "score_threshold": 0.05,
        },
    )
    print(retriever)

    return retriever


def generate_response(retriever, input_line):
    template = """
    Eres el mejor vendedor inmobiliario.
    Contesta la pregunta que se hace basandote unicamente en el contexto que se proporciona.

    <context>
    {context}
    </context>

    Pregunta: {input}
    """
    model = ChatOpenAI(
        temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_template(template)


    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_line})

    return response


def main_app(file_path: str):
    # Load the data
    data = load_data(file_path)
    data_columns = data.columns.tolist()
    docs = create_documents(data, data_columns)
    docs = load_csv_with_loader()
    

    # Create documents from the data
    # ids, docs = create_documents(data, data_columns)

    retriever = get_retriever(docs)

    input_line = "Qué propiedades tienes en Capital Norte y que caracteristicas tienen?"
    response = generate_response(retriever, input_line)
    print(response)

if __name__ == "__main__":
    main_app("listado_propiedades.csv")