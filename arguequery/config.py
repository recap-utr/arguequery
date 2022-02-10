from typing import Any

from dynaconf import Dynaconf

config: Any = Dynaconf(
    envvar_prefix="ARGUEQUERY",
    settings_files=["settings.toml", ".secrets.toml"],
    # load_dotenv=True,
)
