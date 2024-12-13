import subprocess
import os
import pandas as pd
from typing import List, Dict, Tuple
import time
import json

# Define paths
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

# Analysis categories
ANALYSIS_CATEGORIES = {
    "Emotional_Analysis": [
        "initial_emotional_state",
        "current_emotional_state",
        "emotional_transformation",
        "internal_conflict"
    ],
    "Context_Reconstruction": [
        "immediate_situation",
        "leading_circumstances",
        "moment_significance",
        "external_factors"
    ],
    "Historical_Background": [
        "past_events",
        "personal_history",
        "relationships_referenced",
        "perspective_changes"
    ],
    "Behavioral_Analysis": [
        "behavior_emotional_indicators",
        "contradictory_actions",
        "goals_avoidance",
        "coping_mechanisms"
    ],
    "Future_Implications": [
        "future_behavior_impact",
        "unresolved_emotions",
        "potential_growth",
        "future_decisions"
    ]
}

def read_input_texts(filepath: str = "diverse_emotion_text.txt") -> List[str]:
    """Read input texts from file"""
    try:
        with open(filepath, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Successfully loaded {len(texts)} texts from {filepath}")
        return texts
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []

def create_modelfile(model_name: str) -> str:
    """Create modelfile for complex emotion analysis"""
    modelfile_content = f'''FROM {model_name}
SYSTEM """You are an advanced emotion analyzer. For each text:
1. Emotional Analysis:
   - Initial emotional state
   - Current emotional state
   - Emotional transformation
   - Internal conflicts

2. Context Analysis:
   - Immediate situation
   - Circumstances leading to this moment
   - Significance of the moment
   - External influences

3. Historical Analysis:
   - Past events impact
   - Personal history relevance
   - Key relationships
   - Perspective changes

4. Behavioral Patterns:
   - Emotional indicators in behavior
   - Contradictory actions analysis
   - Goals and avoidances
   - Coping mechanisms

5. Future Implications:
   - Potential impact on future behavior
   - Unresolved emotional aspects
   - Growth opportunities
   - Future decision points

Provide analysis in clear sections with detailed insights."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER presence_penalty 0.6'''

    modelfile_path = os.path.abspath("modelfile.txt")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    return modelfile_path

def run_emotion_analysis(text: str, model_name: str) -> Dict:
    """Run complex emotion analysis"""
    command = [
        "apptainer", "run", "--nv",
        "--bind", f"{OLLAMA_DATA}:/ollama_data",
        "--env", "OLLAMA_MODELS=/ollama_data",
        "--env", "CUDA_VISIBLE_DEVICES=0,1",
        os.path.join(OLLAMA_PATH, "ollama.sif"),
        "run", model_name,
        f"""Analyze this text for complex emotional patterns:

Text: {text}

Provide a detailed analysis covering:
1. Emotional Analysis
- What is the initial emotional state described?
- What is the current emotional state?
- What emotional transformation occurred?
- What internal conflict is present?

2. Context Reconstruction
- What is the immediate situation being described?
- What circumstances led to this moment?
- Why is this moment significant to the person?
- What external factors are influencing their behavior?

3. Historical Background
- What past events likely led to this situation?
- What personal history shapes their current response?
- What relationships or experiences are being referenced?
- How has their perspective changed over time?

4. Behavioral Analysis
- What does their specific behavior reveal about their emotional state?
- Why are they choosing to act in contradictory ways?
- What are they trying to achieve or avoid?
- What coping mechanisms are they displaying?

5. Future Implications
- How might this situation affect their future behavior?
- What unresolved emotions need to be addressed?
- What potential growth or change might occur?
- What decisions might they face moving forward?

Structure your response with clear headers and detailed explanations."""
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error: {stderr}")
            return {
                "sections": {},
                "confidence": 0,
                "error": stderr
            }

        # Parse sections from the response
        sections = {}
        current_section = None
        current_content = []
        
        for line in stdout.strip().split('\n'):
            if any(section in line for section in ["Emotional Analysis", "Context Reconstruction", 
                                                 "Historical Background", "Behavioral Analysis", 
                                                 "Future Implications"]):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

         section_confidences = {}
         for section_name, content in sections.items():
             section_confidences[section_name] = calculate_section_confidence(
                content, section_name.replace(" ", "_")
            )

        return {
            "sections": sections,
            "confidence_scores": section_confidences,
            "response": stdout.strip()
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "sections": {},
            "confidence": 0,
            "error": str(e)
        }

def calculate_section_confidence(section_text: str, section_type: str) -> Dict[str, float]:
    """Calculate confidence scores for each aspect of a section"""
    
    confidence_metrics = {
        "Emotional_Analysis": {
            "clarity": len(section_text.split()) > 50,  # Longer responses indicate more detail
            "specificity": any(word in section_text.lower() for word in 
                             ["specifically", "particularly", "notably"]),
            "transformation": "transform" in section_text.lower(),
            "conflict": "conflict" in section_text.lower()
        },
        "Context_Reconstruction": {
            "detail": len(section_text.split()) > 40,
            "causality": any(word in section_text.lower() for word in 
                           ["because", "due to", "as a result"]),
            "timeline": any(word in section_text.lower() for word in 
                          ["before", "after", "during", "when"])
        },
        "Historical_Background": {
            "temporal": any(word in section_text.lower() for word in 
                          ["past", "previously", "earlier", "before"]),
            "relationship": any(word in section_text.lower() for word in 
                              ["relationship", "connection", "interaction"]),
            "development": len(section_text.split()) > 45
        },
        "Behavioral_Analysis": {
            "motivation": any(word in section_text.lower() for word in 
                            ["because", "motivation", "reason"]),
            "pattern": any(word in section_text.lower() for word in 
                         ["pattern", "consistently", "tends to"]),
            "mechanism": "mechanism" in section_text.lower()
        },
        "Future_Implications": {
            "prediction": any(word in section_text.lower() for word in 
                            ["might", "could", "would", "likely"]),
            "growth": "growth" in section_text.lower(),
            "resolution": any(word in section_text.lower() for word in 
                            ["resolve", "address", "overcome"])
        }
    }

    if section_type not in confidence_metrics:
        return {"overall": 50.0}  # Default confidence

    # Calculate section-specific confidence
    metrics = confidence_metrics[section_type]
    scores = [100 if metric else 50 for metric in metrics.values()]
    
    # Calculate specific aspect confidences
    aspect_confidence = {
        aspect: 100 if value else 50
        for aspect, value in metrics.items()
    }
    
    # Add overall confidence
    aspect_confidence["overall"] = sum(scores) / len(scores)
    
    return aspect_confidence

def save_all_results(all_results: List[Dict], test_texts: List[str], selected_models: List[str]):
    """Save results in both CSV and markdown formats"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed markdown analysis
    with open(f"mixed_emotions_comparison_{timestamp}.md", 'w') as f:
        f.write("# Complex Emotion Analysis Results\n\n")
        for result in all_results:
            f.write(f"## Text: {result['text']}\n")
            f.write(f"Model: {result['model']}\n\n")
            for section, content in result['analysis']['sections'].items():
                f.write(f"### {section}\n")
                f.write(f"{content}\n\n")
                
                # Add confidence scores
                conf_scores = result['analysis']['confidence_scores'].get(section, {})
                f.write("#### Confidence Scores:\n")
                for aspect, score in conf_scores.items():
                    f.write(f"- {aspect}: {score:.1f}%\n")
                f.write("\n")
            f.write("-" * 70 + "\n\n")


    # Create DataFrame for CSV
    df_results = []
    for result in all_results:
        df_results.append({
            'text': result['text'],
            'model': result['model'],
            'full_analysis': result['analysis']['response'],
            'confidence': result['analysis']['confidence']
        })
    
    # Save CSV
    df_all = pd.DataFrame(df_results)
    df_all.to_csv(f"mixed_emotions_comparison_{timestamp}.csv", index=False)
    
    # Print comparison
    print("\nModel Comparison:")
    print("=" * 70)
    for text in test_texts:
        print(f"\nText: {text}")
        for model in selected_models:
            result = df_all[(df_all['text'] == text) & (df_all['model'] == model)].iloc[0]
            print(f"\n{model}:")
            print(f"Analysis:\n{result['full_analysis']}")
            print(f"Confidence: {result['confidence']}%")
        print("-" * 70)

if __name__ == "__main__":
    # Available models
    models = [
        "llama3.1:8b",     # Llama 3.1 8B model
        "llama3.1:70b",    # Llama 3.1 70B model
        "llama3.2:3b",     # Llama 3.2 3B model
        "llama3.2:1b",     # Llama 3.2 1B model
        "mistral",         # Mistral model
        "qwen2.5-coder",   # Qwen 2.5-coder model
        "qwq"              # qwq model
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Get user input for model selection
    print("\nSelect models to run (comma-separated numbers, e.g., '1,2,3' for all):")
    selections = input().split(',')
    selected_models = [models[int(s.strip())-1] for s in selections]
    
    # Read input texts
    test_texts = read_input_texts()
    if not test_texts:
        exit(1)

    # Process each text with selected models
    all_results = []
    for text in test_texts:
        for model in selected_models:
            print(f"\nProcessing text with {model}...")
            analysis = run_emotion_analysis(text, model)
            all_results.append({
                "text": text,
                "model": model,
                "analysis": analysis
            })
            time.sleep(1)  # Prevent overloading

    # Save all results in both formats
    save_all_results(all_results, test_texts, selected_models)
    print("\nAnalysis complete. Results saved in both CSV and markdown formats.")
