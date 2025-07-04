import requests
import subprocess
import time

OLLAMA_URL = "http://localhost:11434"

def _is_ollama_running():
    """
    Check if the Ollama server is currently running by pinging the tags endpoint.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def _start_ollama_server():
    """
    Attempt to start the Ollama server in the background, and wait until it's ready.
    """
    print("Starting the Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait up to 10 seconds for the server to come online
    for _ in range(10):
        time.sleep(1)
        if _is_ollama_running():
            return True
    return False

def start_ollama():
    """
    Ensure Ollama is running, starting it if necessary.
    """
    if not _is_ollama_running():
        _start_ollama_server()

def is_model_downloaded(model_name):
    """
    Check if a specific Ollama model is downloaded and available.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        return model_name in model_names
    except requests.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return False
