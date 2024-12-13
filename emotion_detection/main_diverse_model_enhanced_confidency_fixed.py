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
    modelfile_content = f'''FROM {model_name}
SYSTEM """You are an emotion classifier. For each text:
1. Identify the primary emotion (joy, sadness, anger, fear, love, surprise)
2. Provide a realistic confidence score (20-85%). Consider:
   - Ambiguous emotions should have lower confidence (20-50%)
   - Mixed emotions should reduce confidence by 20%
   - Clear, single emotions can have higher confidence (50-85%)
   Never give 100% confidence as human emotions are complex.
Respond exactly as: 'Emotion: [emotion] (Confidence: [score]%)'"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER presence_penalty 0.6'''

    modelfile_path = os.path.abspath("modelfile.txt")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    return modelfile_path


def run_emotion_analysis(text: str, model_name: str) -> Tuple[str, int]:
    command = [
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        "--env", "CUDA_VISIBLE_DEVICES=0,1",  # Use multiple GPUs
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", model_name,
        f"""Analyze this text considering context, cultural nuances, and emotional complexity. 
            Identify the primary emotion and provide a realistic confidence score.
            Text: {text}"""
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error: {stderr}")
            return "Error", 0
            
        response = stdout.strip()
        
        try:
            emotion = response.split('Emotion: ')[1].split(' (Confidence:')[0]
            confidence = int(response.split('Confidence: ')[1].split('%')[0])
            
            # Apply confidence adjustment
            if '/' in emotion or ',' in emotion:  # Mixed emotions
                confidence = min(confidence, 60)
            confidence = min(max(confidence, 20), 85)  # Enforce limits
            
            return emotion.strip(), confidence
        except:
            return response, 0
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error", 0


def analyze_results(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)

    summary = pd.DataFrame({
        'model': df['model'].unique(),
        'avg_confidence': df.groupby('model')['confidence'].mean(),
        'std_confidence': df.groupby('model')['confidence'].std(),
        'most_common_emotion': df.groupby('model')['emotion'].agg(lambda x: x.mode()[0]),
        'emotion_diversity': df.groupby('model')['emotion'].nunique()
    })

    summary.to_csv("model_analysis.csv")
    return summary


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
        "llama3.2:3b",     # Llama 3.2 3B model
        "llama3.2:1b",     # Llama 3.2 1B model
        "mistral",         # Mistral model
        "qwen2.5-coder"
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
