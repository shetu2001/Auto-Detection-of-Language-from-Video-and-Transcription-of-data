import whisperx
import ffmpeg
import os
import torch
import torchaudio
import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- Step 1: Extract Audio ----------
def extract_audio(video_path, output_audio_path):
    try:
        # Set quiet=False to see ffmpeg output and capture_stderr=True
        out, err = ffmpeg.input(video_path).output(output_audio_path, ac=1, ar='16000').run(capture_stderr=True, overwrite_output=True, quiet=False)
        return output_audio_path
    except ffmpeg.Error as e:
        # Check if stdout is not None before decoding
        if e.stdout:
            print('ffmpeg stdout:', e.stdout.decode('utf8'))
        else:
            print('ffmpeg stdout: (No output)')
        # Check if stderr is not None before decoding
        if e.stderr:
            print('ffmpeg stderr:', e.stderr.decode('utf8'))
        else:
            print('ffmpeg stderr: (No output)')
        raise # Re-raise the exception after printing the details


# # ---------- Step 2: Transcribe and Detect Language ----------

def transcribe_auto(audio_path, model_size="medium"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load WhisperX model
    model = whisperx.load_model(model_size, device=device, compute_type='float32')

    # Transcribe audio
    result = model.transcribe(audio_path)
    detected_lang = result['language']
    print(f"Detected language: {detected_lang}")

	# Define languages that support alignment (you can expand this list)
    supported_alignment_langs = {"en", "fr", "de", "es", "it", "pt", "nl"}  # etc.

    if detected_lang == "pa":
        try:
            processor = Wav2Vec2Processor.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
            wav2vec_model = Wav2Vec2ForCTC.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
            speech_array, sampling_rate = torchaudio.load(audio_path)
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                speech_array = resampler(speech_array)
            speech = speech_array.squeeze().numpy()
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = wav2vec_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            return transcription,detected_lang
        except Exception as e:
            return None,detected_lang
    elif detected_lang == "bn":
        try:
            from banglaspeech2text import Speech2Text
            stt = Speech2Text("base")
            transcription = stt.recognize(audio_path)
            return transcription,detected_lang
        except Exception as e:
            return None,detected_lang

    # Perform alignment only if language is supported
    elif detected_lang in supported_alignment_langs:
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        aligned_result = whisperx.align(result['segments'], model_a, metadata, audio_path, device=device)
        return aligned_result['segments'], detected_lang
    else:
        print(f"Alignment not supported for language: {detected_lang}. Returning segment-level transcription only.")
        return result['segments'], detected_lang


# ---------- Step 3: Save Transcript ----------
def save_transcript(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        if isinstance(segments, str):
            f.write(segments + "\n")
        else:
            for seg in segments:
                f.write(seg['text'] + "\n")


# ---------- Step 4: Process Multiple Videos ----------
def process_videos(video_folder, output_folder, model_size="medium"):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(video_folder):
        if file_name.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(video_folder, file_name)
            audio_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + "_audio.wav")
            print(audio_path)
            print(f"\nProcessing: {file_name}")
            t1 = time.time()
            extract_audio(video_path, audio_path)
            t2 = time.time()
            segments, lang = transcribe_auto(audio_path, model_size)
            t3 = time.time()
            transcript_file = os.path.join(output_folder, f"{file_name.rsplit('.', 1)[0]}_transcript_{lang}.txt")
            save_transcript(segments, transcript_file)
            t4 = time.time()
            print(f"Transcript saved for {file_name} in detected language: {lang}\n")
            print(f"Time taken: {t2-t1:.2f}s to extract audio, {t3-t2:.2f}s to transcribe, {t4-t3:.2f}s to save transcript.")



# ---------- Main ----------
if __name__ == "__main__":
    # Folder with Indian language videos
    video_folder = "/content/drive/MyDrive/CDAC Project/videos"
    output_folder = "/content/transcripts"

    process_videos(video_folder, output_folder, model_size="medium")
