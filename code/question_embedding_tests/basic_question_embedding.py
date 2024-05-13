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

embedding_model = OpenAIEmbeddings()
model_name = 'gpt-3.5-turbo-0125'
collection_name="taylor-swift-basic"

db = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db", collection_name=collection_name)

retriever = db.as_retriever()


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



