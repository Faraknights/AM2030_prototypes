import subprocess
from flask import Blueprint, request, jsonify
import base64
import os
import whisper
import tempfile
import torch
import requests
import json

folltl_generation_bp = Blueprint('folltl', __name__)
transcribe_bp = Blueprint('transcribe', __name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisp = whisper.load_model("turbo", device=device)

OLLAMA_URL = "http://ollama:11434/api/generate"


def query_ollama(model: str, text: str) -> str:
    """Query a local Ollama model with raw text."""
    payload = {
        "model": model,
        "prompt": text,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Error querying Ollama ({model}):", e)
        return ""


def safe_json_parse(text: str, fallback):
    """Safely parse JSON output."""
    try:
        return json.loads(text.replace("json", "").replace("`", ""))
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed. Raw output:")
        print(text)
        return fallback

@folltl_generation_bp.route('/folltl', methods=['POST'])
def folltl_generation():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        user_text = data["text"]

        step1_raw = query_ollama("preference_step1", user_text)

        step1 = safe_json_parse(step1_raw, [])
        return jsonify({"generated_text": step1}), 200
        if not step1:
            return jsonify({"generated_text": "No preferences detected."}), 200

        step2_input = json.dumps(step1, indent=2)
        step2_raw = query_ollama("preference_step2", step2_input)
        step2 = safe_json_parse(step2_raw, step1)

        step3_input = json.dumps(step2, indent=2)
        logic = query_ollama("preference_step3", step3_input)

        return jsonify({"generated_text": logic}), 200

    except Exception as e:
        return jsonify({"error": "Folltl generation failed", "message": str(e)}), 500

@transcribe_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    tmp_file_path = None
    resampled_path = None

    try:
        data = request.get_json()

        if not data or "encoded_audio" not in data:
            return jsonify({"error": "Missing 'encoded_audio' field"}), 400

        audio_data = base64.b64decode(data["encoded_audio"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name

        # Resample the audio to 16kHz mono WAV
        resampled_path = tmp_file_path.replace(".wav", "_resampled.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_file_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", resampled_path
        ], check=True)

        result = whisp.transcribe(resampled_path, patience=2, beam_size=5)
        transcription = result.get("text", "").strip()

        return jsonify({"transcription": transcription}), 200

    except Exception as e:
        return jsonify({
            "error": "Transcription failed",
            "message": str(e)
        }), 500

    finally:
        for path in [tmp_file_path, resampled_path]:
            if path and os.path.exists(path):
                os.remove(path)
