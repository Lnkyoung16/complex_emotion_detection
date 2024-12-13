import subprocess
import os
import pandas as pd
from typing import List, Dict, Tuple
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datasets import load_dataset
import re

class EmotionDetector:
    def __init__(self, model_name: str = "llama3.1:70b"):
        self.OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
        self.OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"
        self.model_name = model_name
        self.examples = []

    def load_examples(self, n_examples: int = 3):
        dataset = load_dataset("dair-ai/emotion", split="train")
        df = pd.DataFrame(dataset)
        self.examples = []
        for emotion in df['label'].unique():
            examples = df[df['label'] == emotion].sample(min(n_examples, len(df[df['label'] == emotion])))
            self.examples.extend(examples[['text', 'label']].values.tolist())

    def run_emotion_analysis(self, text: str) -> Tuple[str, float]:
        prompt = self._create_prompt(text)
        command = [
            "apptainer", "run", "--nv",
            "--bind", f"{self.OLLAMA_DATA}:/ollama_data",
            "--env", "OLLAMA_MODELS=/ollama_data",
            os.path.join(self.OLLAMA_PATH, "ollama.sif"),
            "run", self.model_name,
            prompt
        ]
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            
            if stderr:
                print(f"Warning: {stderr}")
            return self._parse_response(stdout.strip())
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return "Error", 0.0

    def _create_prompt(self, text: str) -> str:
        prompt = "Analyze the emotion in this text and respond ONLY with the emotion and confidence in this format - Emotion: [emotion] (Confidence: [score]%)\n\n"
        
        if self.examples:
            prompt += "Examples:\n"
            for ex_text, ex_label in self.examples[:3]:
                prompt += f"Text: {ex_text}\nEmotion: {ex_label}\n\n"
        
        prompt += f"Text: {text}"
        return prompt

    def _parse_response(self, response: str) -> Tuple[str, float]:
        try:
            emotion_match = re.search(r'\*\*(.*?)\*\*|Emotion:\s*(.*?)[\n\(]', response)
            emotion = emotion_match.group(1) or emotion_match.group(2) if emotion_match else "unknown"
            
            confidence_match = re.search(r'\*\*(0\.\d+)\*\*|Confidence:\s*(\d+\.?\d*)|(\d+\.?\d*)%', response)
            if confidence_match:
                confidence_str = next(g for g in confidence_match.groups() if g is not None)
                confidence = float(confidence_str)
                if confidence > 1:
                    confidence /= 100
            else:
                confidence = 0.5
                
            return emotion.strip(), confidence
            
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Response was: {response}")
            return "unknown", 0.5

    def evaluate(self, n_samples: int = 100):
        dataset = load_dataset("dair-ai/emotion", split="test")
        df = pd.DataFrame(dataset).sample(n_samples)
        
        true_labels = []
        pred_labels = []
        confidences = []
        
        for idx, row in df.iterrows():
            emotion, confidence = self.run_emotion_analysis(row['text'])
            true_labels.append(row['label'])
            pred_labels.append(emotion.lower())
            confidences.append(confidence)
            time.sleep(1)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{n_samples} samples")
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels),
            'classification_report': classification_report(true_labels, pred_labels),
            'avg_confidence': sum(confidences) / len(confidences)
        }
        
        return metrics, pd.DataFrame({
            'text': df['text'],
            'true_label': true_labels,
            'predicted': pred_labels,
            'confidence': confidences
        })

def main():
    detector = EmotionDetector(model_name="llama3.1:70b")
    detector.load_examples()
    
    test_text = "I just got promoted at work!"
    emotion, confidence = detector.run_emotion_analysis(test_text)
    print(f"\nText: {test_text}")
    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nRunning evaluation...")
    metrics, results_df = detector.evaluate(n_samples=20)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.2f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    results_df.to_csv("emotion_detection_results.csv", index=False)

if __name__ == "__main__":
    main()
