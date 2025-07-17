import os

def load_text(input_str):
    if os.path.isfile(input_str) and input_str.endswith('.txt'):
        with open(input_str, 'r', encoding='utf-8') as f:
            return f.read()
    return input_str  # assume direct text input