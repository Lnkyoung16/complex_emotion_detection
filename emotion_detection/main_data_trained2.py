# main_data_trained.py
import subprocess
import os
import pandas as pd
from typing import List, Dict, Tuple
import time

class EmotionDetector:
    def __init__(self, model_name: str = "llama3.1:70b"):
        self.OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
        self.OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"
        self.model_name = model_name
        print(f"Initialized EmotionDetector with model: {model_name}")

    def run_emotion_analysis(self, text: str) -> Tuple[str, float]:
        """Run emotion analysis with debug output"""
        prompt = (
            "Analyze the emotion in this text and respond ONLY with the emotion "
            "and confidence in this exact format - Emotion: [emotion] (Confidence: [score]%)\n\n"
            f"Text: {text}"
        )
        
        print(f"\nAnalyzing text: {text}")
        command = [
            "apptainer", "run", "--nv",
            "--bind", f"{self.OLLAMA_DATA}:/ollama_data",
            "--env", "OLLAMA_MODELS=/ollama_data",
            os.path.join(self.OLLAMA_PATH, "ollama.sif"),
            "run", self.model_name,
            prompt
        ]
        
        try:
            print("Executing command...")
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if stderr:
                print(f"Warning: {stderr}")
            
            if stdout:
                print(f"Response: {stdout.strip()}")
                return self._parse_response(stdout.strip())
            else:
                print("No output received")
                return "Error", 0.0
                
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return "Error", 0.0

    def _parse_response(self, response: str) -> Tuple[str, float]:
        """Parse model response with debug output"""
        try:
            print(f"Parsing response: {response}")
            if 'Emotion:' in response:
                parts = response.split('(Confidence:')
                emotion = parts[0].replace('Emotion:', '').strip()
                confidence = float(parts[1].strip('%)')) / 100.0
                print(f"Parsed - Emotion: {emotion}, Confidence: {confidence}")
                return emotion, confidence
            else:
                print("Response format not recognized")
                return response, 0.5
        except Exception as e:
            print(f"Parsing error: {e}")
            return "Error parsing", 0.5

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

def main():
    print("\nStarting emotion detection...")
    detector = EmotionDetector(model_name="llama3.1:70b")
    
    test_texts = read_input_texts()
    
    print("\nTesting individual texts:")
    results = []
    
    for text in test_texts:
        print(f"\n{'='*50}")
        emotion, confidence = detector.run_emotion_analysis(text)
        results.append({
            'text': text,
            'emotion': emotion,
            'confidence': confidence
        })
        print(f"Result - Emotion: {emotion}, Confidence: {confidence:.2%}")
        time.sleep(1)  # Rate limiting
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("emotion_detection_results.csv", index=False)
    print("\nResults saved to emotion_detection_results.csv")

if __name__ == "__main__":
    main()
