# main.py
import subprocess
import os
import pandas as pd
from typing import List, Dict
import time

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

def create_modelfile(model_name: str) -> str:
    """Create modelfile and return its path"""
    modelfile_content = f'''FROM {model_name}
SYSTEM "You are an emotion classifier. When given text, respond with exactly one word from this list: joy, sadness, anger, fear, love, surprise. No other words or explanations."
PARAMETER temperature 0.3'''
    
    modelfile_path = os.path.abspath("modelfile.txt")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    return modelfile_path

def run_emotion_analysis(text: str, model_name: str) -> str:
    """Run single emotion analysis"""
    command = [
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", model_name,
        f"Classify the emotion in this text as one of: joy, sadness, anger, fear, love, surprise. Respond with only the emotion word. Text: {text}"
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running analysis: {stderr}")
            return "Error"
        
        return stdout.strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error"

def batch_analyze(texts: List[str], model_name: str) -> List[Dict]:
    """Analyze multiple texts and return results"""
    results = []
    total = len(texts)
    
    print(f"\nProcessing {total} texts with {model_name}...")
    print("-" * 50)
    
    for i, text in enumerate(texts, 1):
        emotion = run_emotion_analysis(text, model_name)
        results.append({
            'text': text,
            'emotion': emotion,
            'model': model_name
        })
        print(f"[{i}/{total}] {text}")
        print(f"Emotion: {emotion}")
        print("-" * 50)
        time.sleep(1)  # Add small delay between requests
    return results

def setup_and_run_model(model_name: str, test_texts: List[str]) -> List[Dict]:
    """Setup and run analysis for a single model"""
    print(f"\n{'='*20} Processing {model_name} {'='*20}")
    
    # Create modelfile
    modelfile_path = create_modelfile(model_name)
    print(f"Created modelfile at: {modelfile_path}")
    
    # Create the model
    print(f"\nCreating emotion detector from {model_name}...")
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "create", f"emotion-detector-{model_name.replace(':', '-')}", "-f", modelfile_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stderr:
        print(f"Warning during creation: {stderr.decode()}")
    
    # Run analysis
    results = batch_analyze(test_texts, model_name)
    
    # Save individual model results
    df = pd.DataFrame(results)
    filename = f"emotion_results_{model_name.replace(':', '-')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    # Available models in your system
    models = [
        "llama3.1:8b",    # Your Llama 3.1 8B model
        "llama3:8b",      # Your Llama 3 8B model
        "mistral"         # Mistral model
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Get user input for model selection
    print("\nSelect models to run (comma-separated numbers, e.g., '1,2,3' for all):")
    selections = input().split(',')
    selected_models = [models[int(s.strip())-1] for s in selections]
    
    # Test texts
    test_texts = [
        "I just got promoted at work!",
        "I miss my family so much it hurts.",
        "I can't believe they would do this to me!",
        "Wow, I never expected this to happen!",
        "I love spending time with you."
    ]
    
    # Run analysis for each selected model
    all_results = []
    for model in selected_models:
        results = setup_and_run_model(model, test_texts)
        all_results.extend(results)
    
    # Save combined results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("emotion_results_comparison.csv", index=False)
    print("\nCombined results saved to emotion_results_comparison.csv")
    
    # Print comparison
    print("\nModel Comparison:")
    print("=" * 60)
    for text in test_texts:
        print(f"\nText: {text}")
        for model in selected_models:
            emotion = df_all[(df_all['text'] == text) & (df_all['model'] == model)]['emotion'].iloc[0]
            print(f"{model}: {emotion}")
        print("-" * 60)
