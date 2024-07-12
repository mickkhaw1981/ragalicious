import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_qdrant.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

# ---- ENV VARIABLES ---- # 
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_CLOUD_KEY = os.environ.get("QDRANT_CLOUD_KEY")

# -- RETRIEVAL -- #

# Define embedding model
base_embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Use a Qdrant VectorStore to embed and store our data
from_cloud_qdrant = Qdrant.from_existing_collection(
    embedding=base_embeddings_model,
    # 3 vector indices - recipe_descriptions, recipe_nutrition, recipe_ingredients
    collection_name="recipe_descriptions",
    url='https://30591e3d-7092-41c4-95e1-4d3c7ef6e894.us-east4-0.gcp.cloud.qdrant.io',
    api_key=QDRANT_CLOUD_KEY
)

# Convert retrieved documents to JSON-serializable format
base_retriever = from_cloud_qdrant.as_retriever()


# -- AUGMENTED -- #

# Define the LLM
base_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    openai_api_key=OPENAI_API_KEY, 
    tags=["base_llm"]
)

# Set up a prompt template
base_rag_prompt_template = """\
You are a friendly AI assistant. Using the provided context, please answer the user's question in a friendly, conversational tone. 

Provide the top 3 options if available. For each option, provide the following information:
1. A brief description of the recipe
2. The URL of the recipe
3. The ratings and number of ratings

If you don't know the answer based on the context, say you don't know. 

After providing your answer, always prompt the user for feedback or more questions in order to continue the conversation.

Context:
{context}

Question:
{question}
"""

base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)


# -- GENERATION -- #

# Chat Start Function: Initialize a RAG (Retrieval-Augmented Generation) chain at the start of each chat session.
@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 
    We will build our LCEL RAG chain here, and store it in the user session. 
    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """ 
    base_rag_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": base_rag_prompt | base_llm | StrOutputParser(), "context": itemgetter("context")}
    )
    cl.user_session.set("base_rag_chain", base_rag_chain)
    
    
 # Message Handling Function: Process and respond to user messages using the RAG chain.
@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.
    We will use the LCEL RAG chain to generate a response to the user question.
    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    base_rag_chain = cl.user_session.get("base_rag_chain")
    msg = cl.Message(content="")
    async for chunk in base_rag_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()