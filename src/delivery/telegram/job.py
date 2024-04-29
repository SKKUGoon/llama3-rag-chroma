from telethon import TelegramClient, events

from auth import TelegramAuth
from telehandler import start_message, iden

auth = TelegramAuth("test_session")
client = TelegramClient(auth.session_name, auth.app_id, auth.app_hash)
client.start(bot_token=auth.token)


# Define the wrapper function for each handler functions
# Define handler functions in `telegram_handler.py` for
# better maintenance.
@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    sender = await event.get_sender()
    id_ = iden(sender)

    print(id_, "connected, added to database")
    msg = start_message()

    await event.respond(msg)


@client.on(events.NewMessage())
async def watchman(event):
    """
    Watch all the message.
    :param event:
    :return:
    """
    print(event.message)


@client.on(events.NewMessage())
async def get_context_text(event):
    if event.message.file:
        sender = await event.get_sender()
        id_ = iden(sender)

        if event.message.file.name.endswith('.txt'):
            ctx_id = event.message.file.name
            await event.respond("Processing your text file... please wait")

            path = await event.message.download_media(file='./tmp')
            with open(path, 'r') as file:
                content = file.read()


            await event.respond(f"Successfully uploaded. Here's a brief summary\n\n{jdb.get_current_summary(id_)}")

        else:
            await event.respond("Please send a text file(`.txt`)")


@client.on(events.NewMessage(pattern='/summary'))
async def answer_question(event):
    sender = await event.get_sender()
    id_ = iden(sender)

    if summary is not None:
        await event.respond(summary)
    else:
        await event.respond("Summary has not been created yet")


@client.on(events.NewMessage(pattern='/question'))
async def answer_question(event):
    sender = await event.get_sender()
    id_ = iden(sender)


def main():
    """
    Start the bot client
    """
    print("------ telegram bot starting ------")
    client.run_until_disconnected()