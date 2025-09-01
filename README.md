# Voice Message Summarization

A research project on building an end-to-end pipeline for real-time summarization of long voice messages from communication tools like WeChat. The project leverages LLMs and fine-tuning techniques to generate concise, structured summaries.

## 🚀 Key Features

* **End-to-End Pipeline**: Automatically processes audio input to generate a text summary.
* **Message Type Classification**: Uses a fine-tuned classifier to identify messages as `notice`, `task`, or `chitchat`.
* **Parameter-Efficient Fine-Tuning (PEFT)**: Employs LoRA to efficiently fine-tune a large language model (`mt5-small`) on a small, custom dataset.
* **Custom Prompting**: Uses a conditional prompt based on message type to generate more accurate summaries.
* **Data Processing**: Handles transcription, removes filler words, and converts data to a machine-readable format.

## 🛠️ Project Structure

This project follows a modular structure to facilitate collaboration and maintenance.

```

voice-sum/
├── data/                       \# Contains all data files
│   ├── raw\_audio/              \# Original audio files (e.g., .mp3)
│   └── labels.jsonl            \# The labeled dataset used for training
│
├── asr/                        \# Audio-to-text transcription module
│   └── transcribe.py           \# Uses `faster-whisper` for transcription
│
├── preprocess/                 \# Text preprocessing module
│   └── remove\_fillers.py       \# Removes filler words and handles informal language
│
├── classifier/                 \# Message type classifier module
│   ├── train\_msg\_type.py       \# Training script for the classifier
│   └── inference.py            \# Inference script for the classifier
│
├── summarizer/                 \# Summarization module
│   ├── train\_lora.py           \# LoRA fine-tuning script for the MT5 model
│   └── mt5\_summarize.py        \# Inference script for the summarizer
│
└── run\_pipeline.py             \# Main script to run the entire end-to-end pipeline

````

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone [https://github.com/YourUsername/YourRepository.git](https://github.com/YourUsername/YourRepository.git)
cd YourRepository
````

### 2\. Set up the Conda environment

We recommend using Conda to manage your project dependencies.

```bash
conda create -n voice_project python=3.9
conda activate voice_project
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can generate one:

```bash
pip freeze > requirements.txt
```

### 4\. Install FFmpeg

FFmpeg is required for audio processing.

  * **macOS**: `brew install ffmpeg`
  * **Linux**: `sudo apt update && sudo apt install ffmpeg`
  * **Windows**: [Download from official site](https://ffmpeg.org/download.html) and add to PATH.

## 🚀 Usage

### Step 1: Data Preparation

Place your labeled data in the `data/` directory. Your `labels.jsonl` file must contain `transcript_clean`, `msg_type`, and `summary_ref` fields.

### Step 2: Train the Models

Run the training scripts to fine-tune your classifier and summarizer models.

```bash
# Train the classifier
python classifier/train_msg_type.py

# Train the summarizer
python summarizer/train_lora.py
```

**Note**: The training process will save the models to `clf_out/` and `summarizer/lora_out/`.

### Step 3: Run the End-to-End Pipeline

To test the full pipeline, run the main script.

```bash
python run_pipeline.py
```

This will process the audio file specified in the script and output the predicted message type and the generated summary.

-----

**Authors**: 

-----
