from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore

collection_name = 'tesla'

embedding_model = OpenAIEmbeddings() # or something else
language_model_name = 'gpt-3.5-turbo-0125'

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

db = Chroma( embedding_function=embedding_model, persist_directory="../chroma_db", collection_name=collection_name)

fs = LocalFileStore("./store_location")

store = create_kv_docstore(fs)

full_retriever = ParentDocumentRetriever(
    vectorstore=db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

template = """Answer the following question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI( model_name = language_model_name)


full_chain = (
    {"context": full_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def call_api(query, options, context):

    output = full_chain.invoke(query)

    result = {
        "output": output,
    }

    return result
