from delivery.telegram.auth import TelegramAuth
from ai.rag import RetrieveAugment

from telethon import TelegramClient, events

# Load the RAG system
ra = RetrieveAugment()
ra.load_embedding_model()
ra.load_vectordb()
ra.load_llama()

# Start the telegram client
auth = TelegramAuth("test_session")
client = TelegramClient(auth.session_name, auth.app_id, auth.app_hash)
client.start(bot_token=auth.token)

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    sender = await event.get_sender()
    id_ = f"{sender.username}({sender.last_name} {sender.first_name})"
    print(id_, "connected")

    start_msg = """
⚡️ **Telegram Chat Powered with AI**

Hello, My name is **Iggy**, your productivity chat bot.
I'm currently in `development and testing` status so my answer might not be entirely accurate

[**How it works:**]
➡️ **Regular Chat**: Just like any other chatbot, you can just casually chat with me.

➡️ **Upload specific files, and ask details about that text**: I can answer based on the specific text, and I will refer you to the original source, where I answered from!

[**Files supported**]

✅ `.txt`
❌ `.pdf`
"""
    await event.respond(start_msg)

@client.on(events.NewMessage())
async def watchman(event):
    """
    Watch all the message given to the bot.
    """
    sender = await event.get_sender()
    print(f"{sender.username}({sender.last_name} {sender.first_name})")
    print(event.message)

@client.on(events.NewMessage())
async def get_user_behavior(event):
    if event.message.file:
        # File uploading event
        ...

    else:
        # Regular chat event
        ...

def main():
    """
    Start the bot client
    """
    print("------ telegram bot starting ------")
    client.run_until_disconnected()
