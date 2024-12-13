import subprocess
import os
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"
OUTPUT_DIR = "output"

def setup_logging():
    """Configure logging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/emotion_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_emotion_analysis(text: str) -> Dict[str, str]:
    """Run single emotion analysis with additional context"""
    # First get primary emotion using trained model
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", "emotion-detector",
        f"Classify the emotion in this text as one of: joy, sadness, anger, fear, love, surprise. Respond with only the emotion word. Text: {text}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    primary_emotion = stdout.decode().strip()
    
    if stderr:
        logging.warning(f"Warning: {stderr.decode()}")
    
    # Get intensity analysis using trained model
    process = subprocess.Popen([
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", "emotion-detector",
        f"Rate the intensity of the emotion in this text on a scale of 1-3 (1=mild, 2=moderate, 3=strong). Respond with only the number. Text: {text}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    intensity = stdout.decode().strip()
    
    return {
        'primary_emotion': primary_emotion,
        'intensity': intensity
    }

def process_file(filepath: str) -> List[str]:
    """Load texts from file"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def analyze_text_features(text: str) -> Dict:
    """Analyze basic text features"""
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "has_exclamation": "!" in text,
        "has_question": "?" in text,
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }

def batch_analyze(texts: List[str]) -> List[Dict]:
    """Analyze multiple texts and return results with enhanced analysis"""
    results = []
    total = len(texts)
    
    print(f"\nProcessing {total} texts...")
    print("-" * 50)
    
    for i, text in enumerate(texts, 1):
        # Get emotion analysis from trained model
        analysis = run_emotion_analysis(text)
        
        # Combine with text features
        result = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'emotion': analysis['primary_emotion'],
            'intensity': analysis['intensity'],
            'features': analyze_text_features(text)
        }
        
        results.append(result)
        
        print(f"[{i}/{total}] {text}")
        print(f"Emotion: {analysis['primary_emotion']} (Intensity: {analysis['intensity']})")
        print("-" * 50)
        
        logging.info(f"Processed text {i}/{total}: {analysis['primary_emotion']}")
    
    return results

def generate_visualizations(results: List[Dict], output_dir: str):
    """Generate analysis visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Emotion distribution plot
    plt.figure(figsize=(10, 6))
    emotions = [r['emotion'] for r in results]
    sns.countplot(x=emotions)
    plt.title('Emotion Distribution')
    plt.savefig(f"{output_dir}/emotion_distribution.png")
    plt.close()
    
    # Intensity distribution
    plt.figure(figsize=(10, 6))
    intensities = [int(r['intensity']) for r in results]
    sns.countplot(x=intensities)
    plt.title('Emotion Intensity Distribution')
    plt.xticks([0, 1, 2], ['Mild', 'Moderate', 'Strong'])
    plt.savefig(f"{output_dir}/intensity_distribution.png")
    plt.close()

def generate_report(results: List[Dict]) -> Dict:
    """Generate analysis summary report"""
    emotions = [r['emotion'] for r in results]
    intensities = [int(r['intensity']) for r in results]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "emotion_distribution": {
            emotion: emotions.count(emotion)
            for emotion in set(emotions)
        },
        "intensity_distribution": {
            "mild": intensities.count(1),
            "moderate": intensities.count(2),
            "strong": intensities.count(3)
        },
        "average_text_length": sum(r['features']['length'] for r in results) / len(results)
    }

def save_results(results: List[Dict], output_dir: str):
    """Save results in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/emotion_results_{timestamp}.csv", index=False)
    
    with open(f"{output_dir}/emotion_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary report
    report = generate_report(results)
    with open(f"{output_dir}/analysis_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults and report saved to {output_dir}/")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logging.info("Starting emotion detection analysis")
    
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
        logging.info(f"Loading texts from {input_file}")
        print(f"Loading texts from {input_file}")
    else:
        texts = test_texts
        logging.info("Using default test cases")
        print("Using default test cases")
    
    try:
        # Run analysis
        results = batch_analyze(texts)
        
        # Generate visualizations
        generate_visualizations(results, OUTPUT_DIR)
        
        # Save all results
        save_results(results, OUTPUT_DIR)
        
        logging.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise
