
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.globals import set_debug

embedding_model = OpenAIEmbeddings()
model_name = 'gpt-3.5-turbo-0125'
collection_name="taylor-swift"

db = Chroma(collection_name=collection_name, embedding_function=embedding_model, persist_directory="./chroma_db")

store = LocalFileStore("./filestore") # The storage layer for the parent documents
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=db,
    byte_store=store,
    id_key=id_key,
)


query = "Someone splashed wine on my t-shirt. Should i confront this person?" # maroon


set_debug(True)


template = """You are Taylor Swift. 
A person, who seeks emotional guidence asks for your help. 
Tell this person exactly what he or she needs to do to resolve his/her issues. 
Do mention your song's title and that listening to it will help the person.
Use a passage from the song to support your advice.
Answer the Question only using the context you are provided with.:

{context}

[Question]: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model_name = model_name)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


def call_api(query, options, context):

    output = chain.invoke(query)

    result = {
        "output": output,
    }
    return result





