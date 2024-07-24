import os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from twilio.rest import Client


class TwilioInputSchema(BaseModel):
    text: str = Field(description="The SMS text body")
    number: str = Field(
        description="The full phone number to send the SMS text to including area code. Example: +12038484755"
    )


@tool("send-text-tool", args_schema=TwilioInputSchema, return_direct=True)
def send_text_tool(text: str, number: str) -> int:
    """Send an SMS text to a specific number"""

    print("SENDING text ", text, number)
    tw_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    message = tw_client.messages.create(
        from_="+18332830457",
        body=text,
        to=number,
    )
    return message.sid
