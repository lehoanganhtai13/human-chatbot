def check_bool(text: str) -> bool:
    if text.lower() == "true":
        return True
    elif text.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid boolean value: {text}")
