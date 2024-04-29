def start_message() -> str:
    return "Hello I am your bot.\nMind you I'm still the test version"


def message_logger():
    ...


def iden(sender: any) -> str:
    username = sender.username if sender.username else "No username"
    first_name = sender.first_name if sender.first_name else "No first name"
    last_name = sender.last_name if sender.last_name else "No last name"

    return f"{username} ({last_name} {first_name})"
