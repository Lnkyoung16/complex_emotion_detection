import streamlit as st
import pandas as pd
import subprocess
import os
import time
from typing import List, Dict
import plotly.express as px
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

Provide analysis in clear sections with detailed insights."""'''
    
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
            st.error(f"Error: {stderr}")
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

        return {
            "sections": sections,
            "confidence": 75,
            "response": stdout.strip()
        }

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {
            "sections": {},
            "confidence": 0,
            "error": str(e)
        }

def save_analysis_results(results: List[Dict], timestamp: str):
    """Save analysis results to files"""
    # Save markdown
    with open(f"complex_emotion_analysis_{timestamp}.md", 'w') as f:
        f.write("# Complex Emotion Analysis Results\n\n")
        for result in results:
            f.write(f"## Text: {result['text']}\n")
            for analysis in result['analyses']:
                f.write(f"### Model: {analysis['model']}\n")
                if 'sections' in analysis['analysis']:
                    for section, content in analysis['analysis']['sections'].items():
                        f.write(f"#### {section}\n{content}\n\n")
            f.write("-" * 70 + "\n\n")

    # Save CSV
    df_results = []
    for result in results:
        for analysis in result['analyses']:
            df_results.append({
                'text': result['text'],
                'model': analysis['model'],
                'confidence': analysis['analysis'].get('confidence', 0),
                **{f"{section}": content 
                   for section, content in analysis['analysis'].get('sections', {}).items()}
            })
    
    df = pd.DataFrame(df_results)
    df.to_csv(f"emotion_analysis_{timestamp}.csv", index=False)
    return df

def main():
    st.set_page_config(page_title="Complex Emotion Analysis", layout="wide")
    
    # Sidebar
    st.sidebar.title("Model Selection")
    available_models = [
        "llama3.1:8b",
        "llama3.1:70b",
        "llama3.2:3b",
        "llama3.2:1b",
        "mistral",
        "qwen2.5-coder",
        "qwq"
    ]
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        available_models,
        default=["mistral"]
    )

    # Main content
    st.title("Complex Emotion Analysis")

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Text", "File Upload", "Sample Texts"]
    )

    if input_method == "Single Text":
        text_input = st.text_area("Enter text to analyze:", height=150)
        texts = [text_input] if text_input else []

    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file:
            texts = uploaded_file.getvalue().decode().split('\n')
            texts = [t.strip() for t in texts if t.strip()]
            st.write(f"Loaded {len(texts)} texts from file")

    else:  # Sample Texts
        sample_texts = [
            "The photos sat in the box for years. When I finally opened it yesterday, I couldn't stop smiling, but I immediately closed it again.",
            "I practiced the speech a hundred times until it was perfect. Standing here now, I keep intentionally making mistakes.",
            "The scholarship letter came today. I've been tracing my parents' signatures over and over, but haven't signed my own name yet."
        ]
        selected_sample = st.selectbox("Choose a sample text:", sample_texts)
        texts = [selected_sample]

    # Analysis section
    if st.button("Analyze") and texts and selected_models:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results = []

        # Progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_analyses = len(texts) * len(selected_models)
        current = 0

        for text in texts:
            text_results = []
            for model in selected_models:
                progress_text.text(f"Analyzing with {model}...")
                analysis = run_emotion_analysis(text, model)
                text_results.append({
                    "model": model,
                    "analysis": analysis
                })
                current += 1
                progress_bar.progress(current / total_analyses)
                time.sleep(1)
            results.append({
                "text": text,
                "analyses": text_results
            })

        progress_text.text("Analysis complete!")

        # Save results
        df = save_analysis_results(results, timestamp)
        st.success("Results saved to files!")

        # Display results
        for result in results:
            st.write("---")
            st.subheader("Text:")
            st.write(result["text"])

            # Create tabs for each model
            model_tabs = st.tabs(selected_models)
            for tab, analysis in zip(model_tabs, result["analyses"]):
                with tab:
                    if "error" in analysis["analysis"]:
                        st.error(f"Error: {analysis['analysis']['error']}")
                    else:
                        for section, content in analysis["analysis"]["sections"].items():
                            st.subheader(section)
                            st.write(content)
                            st.write("---")

            # Compare results
            if len(selected_models) > 1:
                st.subheader("Model Comparison")
                for section in ANALYSIS_CATEGORIES:
                    if st.checkbox(f"Show {section} Comparison"):
                        comparison_data = []
                        for analysis in result["analyses"]:
                            response_length = len(analysis["analysis"]["sections"].get(section, ""))
                            comparison_data.append({
                                "Model": analysis["model"],
                                "Response Length": response_length,
                            })
                        df_comp = pd.DataFrame(comparison_data)
                        fig = px.bar(df_comp, x="Model", y="Response Length",
                                   title=f"{section} Analysis Length Comparison")
                        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
