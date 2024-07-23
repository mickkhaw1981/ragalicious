from io import BytesIO
import os
from pprint import pprint
import uuid
import chainlit as cl
from chainlit.element import ElementBased
from dotenv import load_dotenv

# modules for audio processing
import httpx
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from openai import AsyncOpenAI

from utils.graph import generate_workflow

client = AsyncOpenAI()

# ---- ENV VARIABLES ---- #
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_CLOUD_KEY = os.environ.get("QDRANT_CLOUD_KEY")
QDRANT_CLOUD_URL = "https://30591e3d-7092-41c4-95e1-4d3c7ef6e894.us-east4-0.gcp.cloud.qdrant.io"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")


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
            label="Ideas for special occasions that are part of a specific cuisine",
            message="What are good Middle Eastern dishes to make for Thanksgiving?",
            icon="/public/occasion4.svg",
        ),
        cl.Starter(
            label="Make something with ingredients you have",
            message="What can I make with pasta, lemon and chickpeas?",
            icon="/public/ingredients4.svg",
        ),
    ]


# This function can be used to rename the 'author' of a message.
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Assistant": "RAGalicious"}
    return rename_dict.get(orig_author, orig_author)


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


# Text-to-Speech Function: Take the text answer generated and convert it to an audio file
@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 2048  # try 4096 or 8192 if getting read timeout error. the bigger the chunk size, the fewer API calls but longer wait time
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"Accept": mime_type, "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
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

    # text_answer = await generate_text_answer(
    #     transcription
    # )  # need to change this to generate answer based on base_rag_chain

    # output_name, output_audio = await text_to_speech(text_answer, audio_mime_type)

    # output_audio_el = cl.Audio(
    #     name=output_name,
    #     auto_play=True,
    #     mime=audio_mime_type,
    #     content=output_audio,
    # )
    # answer_message = await cl.Message(content="").send()
    # answer_message.elements = [output_audio_el]
    # await answer_message.update()
