# complex_emotion_detection

## Project Structure
/gscratch/scrubbed/nlee6/
├── ollama/                    # Ollama container files
│   ├── ollama.def            # Container definition
│   └── ollama.sif            # Built container
│
├── ollama_data/              # Model storage
│   └── storage/              # Model weights & configs
│
└── emotion_detection/        # Our project
    ├── src/
    │   ├── models/
    │   │   └── modelfile.txt # Model configuration
    │   └── utils/
    │       └── data_loader.py
    ├── main.py               # Main script
    └── emotion_results.csv   # Output file