# dataset_integration.py
import pandas as pd
from typing import Dict, List
import numpy as np
from collections import Counter
from datasets import load_dataset

def load_goemotions() -> pd.DataFrame:
    """Load GoEmotions dataset"""
    try:
        dataset = load_dataset("go_emotions", 'raw')
        df = pd.DataFrame(dataset['train'])
        print(f"GoEmotions columns: {df.columns.tolist()}")
        print(f"Sample data: {df.iloc[0]}")
        return df
    except Exception as e:
        print(f"Error loading GoEmotions: {e}")
        return pd.DataFrame()

def load_social_behavior() -> pd.DataFrame:
    """Load Social Behavior Emotions dataset"""
    try:
        dataset = load_dataset("hita/social-behavior-emotions")
        df = pd.DataFrame(dataset['train'])
        print(f"Social Behavior columns: {df.columns.tolist()}")
        print(f"Sample data: {df.iloc[0]}")
        return df
    except Exception as e:
        print(f"Error loading Social Behavior: {e}")
        return pd.DataFrame()

def load_isear() -> pd.DataFrame:
    """Load ISEAR dataset"""
    try:
        dataset = load_dataset("dalopeza98/isear-cleaned-dataset")
        df = pd.DataFrame(dataset['train'])
        # Map numeric labels to emotions
        emotion_map = {
            1: 'joy',
            2: 'fear',
            3: 'anger',
            4: 'sadness',
            5: 'disgust',
            6: 'shame',
            7: 'guilt'
        }
        df['emotion'] = df['label'].map(emotion_map)
        print(f"Loaded ISEAR: {len(df)} samples")
        return df
    except Exception as e:
        print(f"Error loading ISEAR: {e}")
        return pd.DataFrame()

def get_emotion_category(emotion: str) -> str:
    """Map specific emotions to general categories"""
    if pd.isna(emotion):
        return 'other'
        
    if not isinstance(emotion, str):
        return 'other'
        
    emotion_categories = {
        'joy': ['joy', 'amusement', 'approval', 'excitement', 'gratitude', 'pride', 'optimism', 'relief', 'admiration'],
        'sadness': ['sadness', 'disappointment', 'grief', 'remorse', 'shame', 'guilt'],
        'anger': ['anger', 'annoyance', 'disapproval', 'disgust'],
        'fear': ['fear', 'nervousness', 'embarrassment'],
        'love': ['love', 'caring', 'desire'],
        'surprise': ['surprise', 'confusion', 'curiosity', 'realization'],
        'trust': ['trust']
    }
    
    emotion_lower = emotion.lower()
    for category, emotions in emotion_categories.items():
        if emotion_lower in emotions:
            return category
    return 'other'

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text preprocessing"""
    if len(df) == 0:
        return df
        
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.strip()
    return df

def combine_datasets(min_samples: int = 100) -> pd.DataFrame:
    """Combine datasets with emotion granularity"""
    print("Loading and processing datasets...")
    
    # Process ISEAR dataset (starting with just this one)
    dfs = []
    
    isear_df = load_isear()
    if len(isear_df) > 0:
        isear_df['source'] = 'isear'
        isear_df['detailed_emotion'] = isear_df['emotion']
        isear_df['primary_emotion'] = isear_df['emotion'].apply(get_emotion_category)
        dfs.append(isear_df)
        print(f"Processed ISEAR dataset: {len(isear_df)} samples")
    
    if not dfs:
        print("No datasets loaded successfully")
        return pd.DataFrame()
    
    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    combined_df = preprocess_text(combined_df)
    combined_df = combined_df.dropna(subset=['text', 'emotion'])
    combined_df = combined_df.drop_duplicates(subset=['text'])
    
    # Filter emotions with too few samples
    emotion_counts = combined_df['primary_emotion'].value_counts()
    valid_emotions = emotion_counts[emotion_counts >= min_samples].index
    combined_df = combined_df[combined_df['primary_emotion'].isin(valid_emotions)]
    
    return combined_df

def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """Get detailed statistics about the dataset"""
    if len(df) == 0:
        return {"error": "No data available"}
        
    stats = {
        'total_samples': len(df),
        'detailed_emotions': df['detailed_emotion'].value_counts().to_dict(),
        'primary_emotions': df['primary_emotion'].value_counts().to_dict(),
        'source_distribution': df['source'].value_counts().to_dict(),
        'avg_text_length': df['text'].str.len().mean(),
    }
    return stats

if __name__ == "__main__":
    print("Starting dataset integration...")
    
    # Combine datasets
    combined_df = combine_datasets(min_samples=100)
    
    if len(combined_df) == 0:
        print("No data to process. Please check dataset loading.")
        exit(1)
    
    # Get stats
    stats = get_dataset_stats(combined_df)
    
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    
    print("\nPrimary Emotion Distribution:")
    for emotion, count in sorted(stats['primary_emotions'].items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {count}")
    
    print("\nDetailed Emotion Distribution (top 10):")
    for emotion, count in sorted(stats['detailed_emotions'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{emotion}: {count}")
    
    print("\nSource Distribution:")
    for source, count in stats['source_distribution'].items():
        print(f"{source}: {count}")
    
    # Save combined dataset
    output_path = "src/data/combined_emotions_dataset_detailed.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined dataset saved to '{output_path}'")
