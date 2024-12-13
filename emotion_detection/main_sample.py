# main.py
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
    return stdout.decode().strip()

def batch_analyze(texts: List[str]) -> List[Dict]:
    """Analyze multiple texts and return results"""
    results = []
    total = len(texts)
    
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
    test_texts = [
        "I just got promoted at work!",
        "I miss my family so much it hurts.",
        "I can't believe they would do this to me!",
        "Wow, I never expected this to happen!",
        "I love spending time with you."
    ]
    
    results = batch_analyze(test_texts)
    save_results(results, "emotion_results.csv")
