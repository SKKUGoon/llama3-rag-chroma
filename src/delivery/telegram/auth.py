from dotenv import load_dotenv

from typing import Tuple
import os


class TelegramAuth:
    token_key: str = "_TELEGRAM_TOKEN"
    app_id_key: str = "_TELEGRAM_APP_ID"
    app_hash_key: str = "_TELEGRAM_APP_PW"

    def __init__(self, session_name: str, env_path: str = "../.env"):
        load_dotenv(env_path)
        self.token = self._get_token()
        self.app_id, self.app_hash = self._get_app_id_hash()

        self.session_name = session_name

    def _get_token(self) -> str:
        token = os.getenv(self.token_key)
        if token is None:
            raise KeyError(f"no `{self.token_key}` in env")

        return token

    def _get_app_id_hash(self) -> Tuple[int, str]:
        app_id = os.getenv(self.app_id_key)
        app_hash = os.getenv(self.app_hash_key)

        if app_id is None or app_hash is None:
            raise KeyError(f"no `{self.app_id_key}` or `{self.app_hash_key}` in env")

        return int(app_id), app_hash
    