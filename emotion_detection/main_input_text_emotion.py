# main_input_text_emotion.py
import subprocess
import os
import pandas as pd
from typing import List, Dict

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

def run_emotion_analysis(text: str) -> str:
    """Run single emotion analysis"""
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", "emotion-detector",
        f"Classify the emotion in this text as one of: joy, sadness, anger, fear, love, surprise. Respond with only the emotion word. Text: {text}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    if stderr:
        print(f"Warning: {stderr.decode()}")
    return stdout.decode().strip()

def process_file(filepath: str) -> List[str]:
    """Load texts from file"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def batch_analyze(texts: List[str]) -> List[Dict]:
    """Analyze multiple texts and return results"""
    results = []
    total = len(texts)
    
    print(f"\nProcessing {total} texts...")
    print("-" * 50)
    
    for i, text in enumerate(texts, 1):
        emotion = run_emotion_analysis(text)
        results.append({
            'text': text,
            'emotion': emotion
        })
        print(f"[{i}/{total}] {text}")
        print(f"Emotion: {emotion}")
        print("-" * 50)
    return results

def save_results(results: List[Dict], filename: str):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Default test cases
    test_texts = [
        "I just got promoted at work!",
        "I miss my family so much it hurts.",
        "I can't believe they would do this to me!",
        "Wow, I never expected this to happen!",
        "I love spending time with you."
    ]
    
    # Check if input file is provided
    input_file = "input_texts.txt"
    if os.path.exists(input_file):
        texts = process_file(input_file)
        print(f"Loading texts from {input_file}")
    else:
        texts = test_texts
        print("Using default test cases")
    
    # Run analysis
    results = batch_analyze(texts)
    
    # Save results
    save_results(results, "emotion_results.csv")
