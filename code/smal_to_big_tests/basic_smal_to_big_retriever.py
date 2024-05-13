from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

collection_name = 'tesla'
embedding_model = OpenAIEmbeddings() # or something else
language_model_name = 'gpt-3.5-turbo-0125'

db = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db", collection_name=collection_name)


template = """Answer the following question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI( model_name = language_model_name)
retriver = db.as_retriever()

child_chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def call_api(query, options, context):

    output = child_chain.invoke(query)

    result = {
        "output": output,
    }
    return result
