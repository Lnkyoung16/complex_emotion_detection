# main.py
import subprocess
import os
import pandas as pd
from typing import List, Dict, Tuple
import time

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

def read_input_texts(filepath: str = "input_texts_50.txt") -> List[str]:
    """Read input texts from file"""
    try:
        with open(filepath, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Successfully loaded {len(texts)} texts from {filepath}")
        return texts
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Creating sample file...")
        with open(filepath, 'w') as f:
            f.write("""I just got promoted at work!
I miss my family so much it hurts.
I can't believe they would do this to me!
Wow, I never expected this to happen!
I love spending time with you.""")
        print(f"Created sample {filepath}. Please modify it with your texts and run again.")
        return []

def create_modelfile(model_name: str) -> str:
    """Create modelfile and return its path"""
    modelfile_content = f'''FROM {model_name}
SYSTEM """You are an emotion classifier. For each text, provide:
1. The primary emotion (one of: joy, sadness, anger, fear, love, surprise)
2. A confidence score between 0-100
Respond in this exact format: 'Emotion: [emotion] (Confidence: [score]%)'"""
PARAMETER temperature 0.3'''
    
    modelfile_path = os.path.abspath("modelfile.txt")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    return modelfile_path

def run_emotion_analysis(text: str, model_name: str) -> Tuple[str, int]:
    """Run single emotion analysis and return emotion and confidence"""
    command = [
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", model_name,
        f"Analyze this text and respond ONLY with the emotion and confidence score in this exact format - Emotion: [emotion] (Confidence: [score]%). Text: {text}"
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running analysis: {stderr}")
            return "Error", 0
        
        response = stdout.strip()
        
        # Parse emotion and confidence from response
        try:
            emotion = response.split('Emotion: ')[1].split(' (Confidence:')[0]
            confidence = int(response.split('Confidence: ')[1].split('%')[0])
            return emotion, confidence
        except:
            return response, 0  # Return full response if parsing fails
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error", 0

def batch_analyze(texts: List[str], model_name: str) -> List[Dict]:
    """Analyze multiple texts and return results"""
    results = []
    total = len(texts)
    
    print(f"\nProcessing {total} texts with {model_name}...")
    print("-" * 50)
    
    for i, text in enumerate(texts, 1):
        emotion, confidence = run_emotion_analysis(text, model_name)
        results.append({
            'text': text,
            'emotion': emotion,
            'confidence': confidence,
            'model': model_name
        })
        print(f"[{i}/{total}] {text}")
        print(f"Emotion: {emotion} (Confidence: {confidence}%)")
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
    # First, check for input texts
    test_texts = read_input_texts()
    if not test_texts:
        exit(1)
        
    # Available models in your system
    models = [
        "llama3.1:8b",     # Llama 3.1 8B model
        "llama3.1:70b",    # Llama 3.1 70B model
        "mistral"          # Mistral model
    ]
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Get user input for model selection
    print("\nSelect models to run (comma-separated numbers, e.g., '1,2,3' for all):")
    selections = input().split(',')
    selected_models = [models[int(s.strip())-1] for s in selections]
    
    # Run analysis for each selected model
    all_results = []
    for model in selected_models:
        results = setup_and_run_model(model, test_texts)
        all_results.extend(results)
    
    # Save combined results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("emotion_results_comparison.csv", index=False)
    print("\nCombined results saved to emotion_results_comparison.csv")
    
    # Print comparison with confidence scores
    print("\nModel Comparison:")
    print("=" * 70)
    for text in test_texts:
        print(f"\nText: {text}")
        for model in selected_models:
            result = df_all[(df_all['text'] == text) & (df_all['model'] == model)].iloc[0]
            print(f"{model}: {result['emotion']} (Confidence: {result['confidence']}%)")
        print("-" * 70)
