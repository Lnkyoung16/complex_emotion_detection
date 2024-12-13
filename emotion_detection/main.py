# main.py
import subprocess
import os

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

def setup_model():
    """Setup and create Ollama emotion detection model"""
    modelfile_path = os.path.abspath("src/models/modelfile.txt")
    print(f"Using modelfile at: {modelfile_path}")
    
    # Create model using modelfile
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"), 
        "create", "emotion-detector",
        "-f", modelfile_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(f"Error: {stderr.decode()}")

def analyze_emotion(text: str):
    """Run emotion analysis on text"""
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", "emotion-detector",
        f"Analyze the emotion in this text: {text}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(f"Error: {stderr.decode()}")
    return stdout.decode()

if __name__ == "__main__":
    # First check if mistral model exists
    print("Pulling mistral model...")
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "pull", "mistral"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(f"Error: {stderr.decode()}")
    
    # Then setup and test model
    print("\nSetting up emotion detector...")
    setup_model()
    
    print("\nTesting emotion detection...")
    text = "I just got promoted at work!"
    emotion = analyze_emotion(text)
