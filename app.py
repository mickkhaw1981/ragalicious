from io import BytesIO
import os
from pprint import pprint
import uuid
import chainlit as cl
from chainlit.element import ElementBased
from dotenv import load_dotenv

# modules for audio processing
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from openai import AsyncOpenAI

from utils.graph import generate_workflow

client = AsyncOpenAI()

# ---- ENV VARIABLES ---- #
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -- AUGMENTED -- #

# Define the LLM
base_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, tags=["base_llm"], temperature=0)
power_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, tags=["base_llm"])


# Conversation starters for the 1st screen
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Plan your quick daily meals",
            message="Give me ideas for making an easy weeknight dinner that takes less than 25 minutes to prepare",
            icon="/public/meals4.svg",
        ),
        cl.Starter(
            label="Ideas for special occasions",
            message="What are good Middle Eastern dishes to make for Thanksgiving?",
            icon="/public/occasion4.svg",
        ),
        cl.Starter(
            label="Use ingredients you have",
            message="Suggest Spanish recipes that are good for the summer that makes use of tomatoes",
            icon="/public/ingredients4.svg",
        ),
    ]


# Chat Start Function: Initialize a RAG (Retrieval-Augmented Generation) chain at the start of each chat session.
@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session.
    We will build our LCEL RAG chain here, and store it in the user session.
    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    langgraph_chain = generate_workflow(base_llm, power_llm)

    cl.user_session.set("langgraph_chain", langgraph_chain)
    cl.user_session.set("thread_id", str(uuid.uuid4()))


# Message Handling Function: Process and respond to user messages using the RAG chain.
@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.
    We will use the LCEL RAG chain to generate a response to the user question.
    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """

    langgraph_chain = cl.user_session.get("langgraph_chain")
    thread_id = cl.user_session.get("thread_id")
    msg = cl.Message(content="")
    langgraph_config = {"configurable": {"thread_id": thread_id, "cl_msg": msg}}

    async for output in langgraph_chain.astream({"question": message.content}, langgraph_config):
        for key, value in output.items():
            pprint(f"================== Node: '{key}':")

    await msg.send()


# Speech-to-Text Function: Convert audio file to text
@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(model="whisper-1", file=audio_file)
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
        if isinstance(chunk, dict) and "response" in chunk and isinstance(chunk["response"], str):
            await msg.stream_token(chunk["response"])

    return msg.content


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
    input_audio_el = cl.Audio(mime=audio_mime_type, content=audio_file, name=audio_buffer.name)
    await cl.Message(author="You", type="user_message", content="", elements=[input_audio_el, *elements]).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)

    transcription = await speech_to_text(whisper_input)
    print("Transcription: ", transcription)

    langgraph_chain = cl.user_session.get("langgraph_chain")
    thread_id = cl.user_session.get("thread_id")
    msg = cl.Message(content="")
    langgraph_config = {"configurable": {"thread_id": thread_id, "cl_msg": msg}}

    async for output in langgraph_chain.astream({"question": transcription}, langgraph_config):
        for key, value in output.items():
            pprint(f"================== Node: '{key}':")

    await msg.send()
