# 🎲 AI-Powered Roleplay Transcriber

This tool uses [WhisperX](https://github.com/m-bain/whisperx) and [pyannote-audio](https://github.com/pyannote/pyannote-audio) to transcribe and diarize audio recordings of tabletop role-playing game (TTRPG) sessions. It splits the audio into chunks, performs speaker diarization, and generates partial and final transcription files for easy analysis or summarization.

## 🧠 What It Does

- Loads an audio file (e.g. your recorded RPG session)
- Splits it into 10 equal parts
- Transcribes each part using WhisperX (with GPU acceleration)
- Identifies speakers with pyannote-audio
- Saves partial transcripts as it's working
- Merges everything into a clean final transcription

## 📂 Output Files

- Individual transcription files for each chunk (e.g. `ROL_chunk_01.txt`, `ROL_chunk_02.txt`, etc.)
- A final merged transcription: `ROL_diarized.txt`

## 🔧 Requirements

- Python 3.10+
- CUDA-enabled GPU
- [WhisperX](https://github.com/m-bain/whisperx)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- `pydub`, `torch`, and other dependencies (see below)

## 📦 Installation

```bash
pip install whisperx pyannote.audio torch torchvision torchaudio pydub
````

You will also need to set up your HuggingFace token:

```python
HF_TOKEN = "your_huggingface_token_here"
```

## ▶️ Usage

Edit the `AUDIO_FILE` variable to point to your `.wav` or `.m4a` audio file:

```python
AUDIO_FILE = r"C:\path\to\your\file.m4a"
```

Then run the script:

```bash
python transcribe_rpg.py
```

## 🎤 Speaker Setup Tip

At the start of your RPG recording, ask each player to clearly say their name ("Hi, I'm John", etc.). This helps link each speaker segment with the actual player during manual review or postprocessing.

## 💡 Next Steps

You can feed the final transcript into ChatGPT or another LLM to:

* Summarize sessions
* Extract character quotes
* Generate story recaps
* Build campaign notes automatically

