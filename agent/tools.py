from langchain_core.tools import tool, BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# define embeddings, this model must match the existing embedding model used to load original vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT",
)
# define vector store collection w. embeddings
hsk_db = FirestoreVectorStore(
    collection="HSK",
    embedding_service=embeddings
)

HSK_retriever = hsk_db.as_retriever(search_kwargs={"k": 10})

hsk_retriever_description = """You must use this tool if the user asks for HSK words. This tool should be called in parallel. 
    This tool searches the HSK exam curriculum and returns vocabulary relevant to the query. 
    Tool_input should be search terms relating to a topic, for example: {topic=happiness, Tool_input=happy happiness},{topic=buddha, Tool_input=buddha buddhism})."""

hsk_vocab = create_retriever_tool(
    HSK_retriever,
    "search_HSK_vocabulary",
    hsk_retriever_description
)



