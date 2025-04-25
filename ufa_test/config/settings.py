from pathlib import Path
from pydantic_settings import BaseSettings


class Base(BaseSettings):

    # General
    project_dir: Path = Path(__file__).resolve().parent.parent

    #  Model path
    model_path: Path = project_dir / 'model.pkl'

    # Input/Output
    input_data_path: Path = project_dir / "database" / "input_data"
    processed_data_path: Path = project_dir / "database" / "processed_data"

    # Logging
    log_level: str = "DEBUG"
    logs_path: Path = project_dir / "logs"

    class Config:
        env_file: str = f"{Path(__file__).resolve().parent.parent}/.env"
        env_file_encoding: str = 'utf-8'
        extra: str = "allow"


settings = Base()
