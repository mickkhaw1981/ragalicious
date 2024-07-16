import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.messages.ai import AIMessageChunk
from langchain.retrievers import EnsembleRetriever
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
from langchain_core.runnables import RunnableLambda

# ---- ENV VARIABLES ---- # 
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_CLOUD_KEY = os.environ.get("QDRANT_CLOUD_KEY")
QDRANT_CLOUD_URL = 'https://30591e3d-7092-41c4-95e1-4d3c7ef6e894.us-east4-0.gcp.cloud.qdrant.io'
# -- RETRIEVAL -- #

# Define embedding model
base_embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Use a Qdrant VectorStore to embed and store our data
qdrant_descriptions = Qdrant.from_existing_collection(
    embedding=base_embeddings_model,
    # 3 vector indices - recipe_descriptions, recipe_nutrition, recipe_ingredients
    collection_name="recipe_descriptions",
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY
)

qdrant_nutrition = Qdrant.from_existing_collection(
    embedding=base_embeddings_model,
    collection_name="recipe_nutrition",
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY
)

qdrant_ingredients = Qdrant.from_existing_collection(
    embedding=base_embeddings_model,
    collection_name="recipe_ingredients",
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY
)

# Convert retrieved documents to JSON-serializable format
descriptions_retriever = qdrant_descriptions.as_retriever(search_kwargs={"k": 20})
nutrition_retriever = qdrant_nutrition.as_retriever(search_kwargs={"k": 20})
ingredients_retriever = qdrant_ingredients.as_retriever(search_kwargs={"k": 20})

ensemble_retriever = EnsembleRetriever(
    retrievers=[
        descriptions_retriever,
        nutrition_retriever,
        ingredients_retriever,
    ],
    weights=[
        0.5,
        0.25,
        0.25,
])

# -- AUGMENTED -- #

# Define the LLM
base_llm = ChatOpenAI(
    model="gpt-4o", 
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

Subsequently, if asked for the full recipe or instructions, or if the user indicates a preferred options, provide the full recipe or instructions. 

You no longer need to provide a brief description, the URL,the ratings and number of ratings once the user has made a selection.

After providing your answer, always prompt the user for feedback or more questions in order to continue the conversation.

Context:
{context}

Question:
{question}
"""

base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

def retriever_output_handler(documents):
    print("returning total results count: ", len(documents))
    for doc in documents: 
        print(f"""{doc.metadata['_collection_name'].ljust(20)} - {doc.metadata['url']} - """)
    
    return documents

# -- GENERATION -- #

# Conversation starters for the 1st screen
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Plan your daily meals",
            message="Give me ideas for making an easy weeknight dinner.",
            icon="/public/meals.svg",
            ),
        cl.Starter(
            label="Ideas for special occasions",
            message="What are good dishes to make for Rosh Hashanah?",
            icon="/public/occasions.svg",
            ),
        cl.Starter(
            label="Make something with ingredients you have",
            message="What can I make with pasta, lemon and chickpeas?",
            icon="/public/ingredients3.svg",
            )
    ]

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
        {"context": itemgetter("question") | ensemble_retriever |  RunnableLambda(retriever_output_handler) , "question": itemgetter("question")}
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
        if isinstance(chunk, dict) and 'response' in chunk and isinstance(chunk['response'], str):
            await msg.stream_token(chunk['response'])


    await msg.send()
