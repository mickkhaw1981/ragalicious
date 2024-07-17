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

# modules for audio processing
import httpx
from chainlit.element import ElementBased
from io import BytesIO
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ---- ENV VARIABLES ---- # 
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_CLOUD_KEY = os.environ.get("QDRANT_CLOUD_KEY")
QDRANT_CLOUD_URL = 'https://30591e3d-7092-41c4-95e1-4d3c7ef6e894.us-east4-0.gcp.cloud.qdrant.io'
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")


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

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Plan your daily meals",
            message="Give me ideas for making an easy weeknight dinner.",
            icon="",
            ),
        cl.Starter(
            label="Get ready to host occasions",
            message="What are good dishes to make for Rosh Hashanah?",
            icon="",
            ),
        cl.Starter(
            label="Get scrappy with ingredients that you already have",
            message="What can I make with pasta, lemon and chickpeas?",
            icon="",
            )
    ]

# Chat Start Function: Initialize a RAG chain at the start of each chat session.
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

# Speech-to-Text Function: Convert audio file to text
@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return response.text

# Generate Text Answer Function: Take the output of Speech-to-Text and generate a text answer
@cl.step(type="tool")
async def generate_text_answer(transcription):
    base_rag_chain = cl.user_session.get("base_rag_chain")
    msg = cl.Message(content="")
    async for chunk in base_rag_chain.astream(
        {"question": transcription},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        if isinstance(chunk, dict) and 'response' in chunk and isinstance(chunk['response'], str):
            await msg.stream_token(chunk['response'])

    return msg.content

# Text-to-Speech Function: Take the text answer generated and convert it to an audio file
@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 2048
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
    "Accept": mime_type,
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    # make an async HTTP POST request to the ElevenLabs API to convert text to speech and return an audio file
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses
        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"
        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        
        buffer.seek(0)
        return buffer.name, buffer.read()


# ---- AUDIO PROCESSING ---- # 

# Audio Chunk Function: Process audio chunks as they arrive from the user's microphone
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    
    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

# Audio End Function: Process the audio file and generate a response
@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You", 
        type="user_message",
        content="",
        elements=[input_audio_el, *elements]
    ).send()
    
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    
    transcription = await speech_to_text(whisper_input)
    print("Transcription: ", transcription)
    text_answer = await generate_text_answer(transcription) # need to change this to generate answer based on base_rag_chain
    
    output_name, output_audio = await text_to_speech(text_answer, audio_mime_type)
    
    output_audio_el = cl.Audio(
        name=output_name,
        auto_play=True,
        mime=audio_mime_type,
        content=output_audio,
    )
    answer_message = await cl.Message(content="").send()
    answer_message.elements = [output_audio_el]
    await answer_message.update()
