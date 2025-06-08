print("⏳ Importing whisperx...")
import whisperx
print("✅ whisperx imported.")

print("⏳ Importing pyannote.audio Pipeline...")
from pyannote.audio import Pipeline
print("✅ pyannote.audio Pipeline imported.")

print("⏳ Importing pydub.AudioSegment...")
from pydub import AudioSegment
print("✅ pydub imported.")

print("⏳ Importing os...")
import os
print("✅ os imported.")

print("⏳ Importing math...")
import math
print("✅ math imported.")

print("⏳ Importing time...")
import time
print("✅ time imported.")

print("⏳ Importing tempfile...")
import tempfile
print("✅ tempfile imported.")

print("⏳ Importing torch...")
import torch
print("✅ torch imported.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
AUDIO_FILE = r""
HF_TOKEN = ""  # Replace with yours

start_time = time.time()
print("🧠 [1/6] Loading WhisperX model...")
model = whisperx.load_model("large", device="cuda", compute_type="float16")
print("✅ Model loaded.\n")

# Create temporary directory for audio chunks
print("✂️ [2/6] Splitting audio into 10 parts...")
audio = AudioSegment.from_file(AUDIO_FILE)
duration_ms = len(audio)
chunk_duration = duration_ms // 10
temp_dir = tempfile.mkdtemp()
chunks = []

for i in range(10):
    start = i * chunk_duration
    end = (i + 1) * chunk_duration if i < 9 else duration_ms
    chunk = audio[start:end]
    chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.wav")
    chunk.export(chunk_file, format="wav")
    chunks.append((chunk_file, start / 1000.0))  # start time in seconds

print(f"✅ Audio split into {len(chunks)} parts.\n")

# Diarization pipeline
print("🧩 [3/6] Loading diarization model...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

# Transcribe and diarize each part
final_segments = []
base_filename = os.path.splitext(os.path.basename(AUDIO_FILE))[0]

for i, (chunk_path, base_time) in enumerate(chunks):
    part_number = i + 1
    print(f"\n🚀 [4/6] Processing part {part_number}/10...")

    # Transcribe
    print("  🎙️ Transcribing...")
    result = model.transcribe(chunk_path, language="es")
    segments = result["segments"]
    for seg in segments:
        seg["start"] += base_time
        seg["end"] += base_time

    # Diarize
    print("  🧩 Performing diarization...")
    diarization = pipeline(chunk_path)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        for seg in segments:
            if seg["start"] >= base_time + turn.start and seg["end"] <= base_time + turn.end:
                seg["speaker"] = speaker
        print("  🧩 Working...")

    # Immediately save partial txt
    chunk_txt_path = os.path.join(temp_dir, f"{base_filename}_chunk_{part_number:02d}.txt")
    with open(chunk_txt_path, "w", encoding="utf-8") as f:
        f.write(f"--- Part {part_number}/10 ---\n")
        for seg in segments:
            speaker = seg.get("speaker", "Speaker_?")
            f.write(f"[{speaker}] {seg['text'].strip()}\n")
    print(f"  💾 Part {part_number} saved to: {chunk_txt_path}")

    # Accumulate segments in case needed
    final_segments.extend(segments)

# Merge all partial txts
output_file = os.path.splitext(AUDIO_FILE)[0] + "_diarized.txt"
print(f"\n📎 [5/6] Merging partial files into: {output_file}")
with open(output_file, "w", encoding="utf-8") as final_file:
    for i in range(10):
        part_txt_path = os.path.join(temp_dir, f"{base_filename}_chunk_{i+1:02d}.txt")
        print(f"  🔗 Adding: {part_txt_path}")
        with open(part_txt_path, "r", encoding="utf-8") as part_file:
            final_file.write(part_file.read())
            final_file.write("\n")

print(f"\n🎉 [6/6] Process completed in {time.time() - start_time:.2f} seconds.")
print(f"📂 Final file generated: {output_file}")
