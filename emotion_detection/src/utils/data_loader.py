# src/utils/data_loader.py
import pandas as pd
from datasets import load_dataset
from typing import Dict, Tuple

class EmotionDataLoader:
    def load_data(self) -> pd.DataFrame:
        """Load emotion dataset"""
        try:
            # Using datasets library instead of direct parquet reading
            dataset = load_dataset("dair-ai/emotion", split="train")
            df = pd.DataFrame(dataset)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def process_text(self, text: str) -> str:
        """Clean and format text for analysis"""
        return text.strip().replace('\n', ' ')

# main.py
def _parse_response(self, response: str) -> Tuple[str, float]:
    """Parse model response to extract emotion and confidence"""
    try:
        # Look for emotion and confidence in the response
        import re
        
        # Find emotion (in bold or after "Emotion:")
        emotion_match = re.search(r'\*\*(.*?)\*\*|Emotion:\s*(.*?)[\n\(]', response)
        emotion = emotion_match.group(1) or emotion_match.group(2) if emotion_match else "unknown"
        
        # Find confidence (in bold, after "Confidence:", or as a decimal)
        confidence_match = re.search(r'\*\*(0\.\d+)\*\*|Confidence:\s*(\d+\.?\d*)|(\d+\.?\d*)%', response)
        if confidence_match:
            confidence_str = next(g for g in confidence_match.groups() if g is not None)
            confidence = float(confidence_str)
            if confidence > 1:  # If it's a percentage
                confidence /= 100
        else:
            confidence = 0.5
            
        return emotion.strip(), confidence
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Response was: {response}")
        return "unknown", 0.5

def create_prompt(self, text: str) -> str:
    """Create prompt with better formatting"""
    prompt = """Analyze the emotion in this text. Respond with ONLY:
1. The primary emotion (one word)
2. Confidence score (0-100%)
Format: Emotion: [emotion] (Confidence: [score]%)

Text: {text}
""".format(text=text)
    return prompt

def run_emotion_analysis(self, text: str) -> Tuple[str, float]:
    """Run emotion analysis with cleaner output"""
    prompt = self.create_prompt(text)
    
    command = [
        "apptainer", "run", "--nv",
        "--bind", f"{self.OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        os.path.join(self.OLLAMA_PATH, "ollama.sif"),
        "run", self.model_name,
        prompt
    ]
    
    try:
        process = subprocess.Popen(command, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if stderr:
            print(f"Warning: {stderr}")
            
        emotion, confidence = self._parse_response(stdout.strip())
        print(f"\nText: {text}")
        print(f"Emotion: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        return emotion, confidence
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error", 0.0
