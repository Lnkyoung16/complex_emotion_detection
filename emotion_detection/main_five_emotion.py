# main.py
import subprocess
import os
import json
from typing import List, Dict
import pandas as pd

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
        f"Classify the emotion in this text as one of: joy, sadness, anger, fear, love, surprise. Response format - Emotion: [category]. Text: {text}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    if stderr:
        print(f"Warning: {stderr.decode()}")
    return stdout.decode().strip()

def batch_analyze(texts: List[str]) -> List[Dict]:
    """Analyze multiple texts and return results"""
    results = []
    for i, text in enumerate(texts, 1):
        emotion = run_emotion_analysis(text)
        results.append({
            'text': text,
            'emotion': emotion
        })
        print(f"Processed {i}/{len(texts)}: {emotion}")
    return results

def save_results(results: List[Dict], filename: str):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Test set
    test_texts = [
        "I just got promoted at work!",
        "I miss my family so much it hurts.",
        "I can't believe they would do this to me!",
        "Wow, I never expected this to happen!",
        "I love spending time with you."
    ]
    
    print("\nRunning emotion detection on test cases...")
    results = batch_analyze(test_texts)
    
    # Save results
    save_results(results, "emotion_results.csv")
    
    # Display summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Emotion: {result['emotion']}")
        print("-" * 50)
