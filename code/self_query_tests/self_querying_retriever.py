from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

embedding_model = OpenAIEmbeddings()
collection_name = "books"
language_model_name = "gpt-3.5-turbo-0125"
collection_name = "books"
vectorstore  = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model, collection_name=collection_name)

metadata_field_info = [
  AttributeInfo(
    name="name",
    description="The name of the book",
    type="string"
  ),
  AttributeInfo(
    name="author",
    description="The name of the author of the book",
    type="string"
  ),
  AttributeInfo(
    name="first_published",
    description="The year of the first publishing of the book",
    type="integer"
  ),
  AttributeInfo(
    name="genre",
    description="Genre or genres of the book",
    type="string"
  ),
  AttributeInfo(
    name="origin",
    description="The country of origin of the book",
    type="string"
  ),
  AttributeInfo(
    name="rating",
    description="The rating of the book on a scale of 1 to 5",
    type="float"
  ),
]
document_content_description = "A description of the books content, themes, characters and setting."

llm = ChatOpenAI( temperature=0, model_name=language_model_name)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)


def call_api(query, options, context):
    # Fetch relevant documents and join them into a string result.
    documents = retriever.get_relevant_documents(query)
    output = "\n".join(f'{doc.metadata}: {doc.page_content}' for doc in documents)

    result = {
        "output": output,
    }

    return result



