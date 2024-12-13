import streamlit as st
import pandas as pd
import subprocess
import os
import time
from typing import List, Dict
import plotly.express as px



# Define paths (same as main script)
OLLAMA_PATH = "/gscratch/scrubbed/nlee6/ollama"
OLLAMA_DATA = "/gscratch/scrubbed/nlee6/ollama_data/storage"

def run_emotion_analysis(text: str, model_name: str) -> Dict:
    """Run emotion analysis (same as main script)"""
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
2. Context Reconstruction
3. Historical Background
4. Behavioral Analysis
5. Future Implications

Structure your response with clear headers and detailed explanations."""
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return {"error": stderr}
            
        return {"response": stdout.strip()}
            
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="Complex Emotion Analysis", layout="wide")
    
    # Sidebar
    st.sidebar.title("Model Selection")
    available_models = [
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral"
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
                time.sleep(1)  # Prevent overloading
            results.append({
                "text": text,
                "analyses": text_results
            })
        
        progress_text.text("Analysis complete!")
        
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
                        st.markdown(analysis["analysis"]["response"])
            
            # Compare results
            if len(selected_models) > 1:
                st.subheader("Model Comparison")
                comparison_data = []
                for analysis in result["analyses"]:
                    response_length = len(analysis["analysis"].get("response", ""))
                    comparison_data.append({
                        "Model": analysis["model"],
                        "Response Length": response_length,
                    })
                
                df = pd.DataFrame(comparison_data)
                fig = px.bar(df, x="Model", y="Response Length", 
                           title="Response Length Comparison")
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
