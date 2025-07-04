import json
from langchain_core.prompts import ChatPromptTemplate

def load_prompt_from_json(path, key):
    with open(path, "r") as f:
        data = json.load(f)

    messages = [
        (entry["role"], entry["content"])
        for entry in data[key]
    ]

    return ChatPromptTemplate.from_messages(messages)