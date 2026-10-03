from hashlib import sha256


def calculate_content_hash(text: str) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()
