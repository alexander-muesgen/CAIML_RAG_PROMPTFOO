
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma

import os
import openai
 
print(os.environ)


embedding_model = OpenAIEmbeddings()
collection_name = "books"
language_model_name = "gpt-3.5-turbo-0125"
collection_name = "books"
vectorstore  = Chroma(persist_directory="../chroma_db", embedding_function=embedding_model, collection_name=collection_name)

def call_api(query, options, context):
    # Fetch relevant documents and join them into a string result.
    
    documents = vectorstore.similarity_search(query)
    output = "\n".join(f'{doc.metadata}: {doc.page_content}' for doc in documents)

    result = {
        "output": output,
    }

    return result




