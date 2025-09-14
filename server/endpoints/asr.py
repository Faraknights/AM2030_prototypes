import subprocess
from flask import Blueprint, request, jsonify
import base64
import os
import whisper
import tempfile
import torch


from modules.generation_folltl import get_logic_translation, get_logic_translation_dial

folltl_generation_bp = Blueprint('emotion', __name__)
transcribe_bp = Blueprint('transcribe', __name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisp = whisper.load_model("turbo", device=device)

@folltl_generation_bp.route('/folltl', methods=['POST'])
def folltl_generation():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        user_text = data["text"]

        # Example: if dialogue flag is passed
        if data.get("is_dialogue", False):
            result = get_logic_translation_dial(user_text)
        else:
            result = get_logic_translation(user_text)

        # Ensure we return a string, not a list/dict from the pipeline
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            generated_text = result[0]["generated_text"]
        else:
            generated_text = str(result)

        return jsonify({"generated_text": generated_text}), 200

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
