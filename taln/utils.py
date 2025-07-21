import os


def is_text_file(file_path, sample_size=1024):
    """Check if a file is text by examining its content."""
    try:
        with open(file_path, "rb") as f:
            sample = f.read(sample_size)

        # Check for null bytes (common in binary files)
        if b"\x00" in sample:
            return False

        # Try to decode as UTF-8
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    except (IOError, OSError):
        return False


def load_text(input_str):
    # Check if input is a file path
    if os.path.isfile(input_str):
        if is_text_file(input_str):
            with open(input_str, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"File {input_str} appears to be binary, not text")

    # If not a file, assume it's direct text input
    return input_str
