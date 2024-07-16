import chainlit as cl

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
            label="Get scrappy with ingredients that you have",
            message="What can I make with pasta, lemon and chickpeas?",
            icon="",
            )
    ]