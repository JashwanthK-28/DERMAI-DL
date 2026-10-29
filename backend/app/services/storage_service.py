import os
import uuid
from typing import Tuple
from backend.app.config import settings


class StorageService:
    @staticmethod
    def generate_storage_key(filename: str, prefix: str = "") -> str:
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            ext = ".png"
        unique_name = f"{prefix}{uuid.uuid4().hex}{ext}"
        return unique_name

    @staticmethod
    def get_original_path(storage_key: str) -> str:
        return os.path.join(settings.ORIGINALS_DIR, storage_key)

    @staticmethod
    def get_preprocessed_path(storage_key: str) -> str:
        return os.path.join(settings.PREPROCESSED_DIR, storage_key)

    @staticmethod
    def save_bytes(content: bytes, destination_path: str):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(content)
