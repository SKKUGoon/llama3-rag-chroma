from delivery.telegram.auth import TelegramAuth
from ai.rag import RetrieveAugment
from ai.parse import Vectorizer, RagTextSource

from telethon import TelegramClient, events
from langchain_core.documents import Document

from typing import List, Tuple, Final
import time

# Load the Vector Database Uploader
vec = Vectorizer()
vec.load_vectordb()
vec.load_embedding_model()

# Load the RAG system
ra = RetrieveAugment()
ra.load_embedding_model()
ra.load_vectordb()
ra.load_llama()

# Start the telegram client
auth = TelegramAuth("test_session")
client = TelegramClient(auth.session_name, auth.app_id, auth.app_hash)
client.start(bot_token=auth.token)
telegram_magic_word: Final = {"/start"}


@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    sender = await event.get_sender()
    _ = f"{sender.username}({sender.last_name} {sender.first_name})"
    
    # Update user information context
    ra.update_user_info(sender.username)
    
    # Hand out starting message
    start_msg = """
⚡️ **Telegram Chat Powered with AI**

Hello, My name is **Iggy**, your productivity chat bot.
I'm being powered by Meta(Facebook) AI's Llama model.
I'm currently in `development` status so my answer might not be entirely accurate.

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
        if event.message.file.name.endswith('.txt'):
            # Process begins
            wait_msg = handle_response_start()
            time_start = time.time()
            await event.respond(wait_msg)

            # Download and process the file
            path = await event.message.download_media(file='/tmp')
            summary = handle_file(path)
            time_end = time.time()
            await event.respond(summary)

            # Process ended
            end_msg = handle_response_end(time_end - time_start)
            await event.respond(end_msg)
        else:
            fail_msg = handle_unsupported_file(event.message.file.name)
            await event.respond(fail_msg)

    elif event.message.message in telegram_magic_word:
        # Skip the rag. Magic Keyword func is handling it
        ...

    else:
        # Regular chat event
        user_q = str(event.message.message)
        wait_msg = handle_response_start()
        await event.respond(wait_msg)

        # Answer and give source + Answering time 
        time_start = time.time()
        answer, source = ra.augmented_generate(question=user_q)

        answer_msg = handle_response_answer(answer)
        source_msg = handle_response_source(source)

        await event.respond(answer_msg)
        await event.respond(source_msg)
        
        time_end = time.time()
        end_msg = handle_response_end(time_end - time_start)
        await event.respond(end_msg)


def handle_file(path: str):
    # TODO: Change the testing title
    collection_test = "test_collection"
    title_test = "telegram testing"
    data = RagTextSource(path, title_test)

    vec_result = vec.vectorify_text(data)
    vec.collection_insert_text(vec_result, data.title, collection_test)

    msg = ""
    with open(path, 'r') as file:
        content = file.read()

        summary = ra.generate_summary(content)
        msg = handle_response_summary(summary)
    return msg



def handle_unsupported_file(filename: str):
    return f"❌ {filename} has unsupported extension. Process terminating."


def handle_response_start():
    return "⏰ Processing..."


def handle_response_end(t: int):
    return f"⏰ Process ended. It took {'%.0f' % round(t, 0)} seconds."


def handle_response_summary(summary: str):
    return f"⚡️ **[Summary]**\n{summary}"


def handle_response_answer(answer: str):
    return f"⚡️ **[Answer]**\n{answer}"


def handle_response_source(source: List[Tuple[Document, float]]):
    source_ls: List[str] = list()
    for doc, relev in source:
        seg = f"➡️[Relevance: {'%.2f' % round(relev * 100, 2)}%] (...){doc.page_content}(...)"
        source_ls.append(seg)

    sources = "\n".join(source_ls)
    msg = f"🎟️ **[Source for the answer]**\n{sources}"
    return msg


def main():
    """
    Start the bot client
    """
    print("\n------ telegram bot starting ------\n")
    client.run_until_disconnected()
 