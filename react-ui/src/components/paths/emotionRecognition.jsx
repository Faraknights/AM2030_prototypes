import { useEffect, useRef, useState } from "react";
import Accordion from "../accordion";

const labels = {
  title: "First-order logic transcription",
  full: "This module will analyze the selected audio file and display the transcription along with the server response.",
  button: "Analyze Audio",
};

const EmotionRecognition = ({ audioFiles, selectedAudio }) => {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const resultRef = useRef(null);

  useEffect(() => {
    if (resultRef.current) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [result]);

  const isAudioValid =
    audioFiles &&
    selectedAudio !== null &&
    audioFiles[selectedAudio] &&
    audioFiles[selectedAudio].transcription &&
    audioFiles[selectedAudio].transcription.trim() !== "";

  const handleAnalyze = async () => {
    setResult(null);
    setStatus(null);

    if (!isAudioValid) {
      setResult({ transcription: "", output: "No audio transcription available." });
      setStatus(400);
      return;
    }

    setIsLoading(true);

    try {
      const transcription = audioFiles[selectedAudio].transcription;

      // Send transcription
      const response = await fetch("https://uncombustible-collinearly-corene.ngrok-free.dev/asr/folltl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcription }),
      });

      const data = await response.json();
      const output = data.generated_text || data.error || "No result received";

      setResult({ transcription, output: output});
      setStatus(response.status);
    } catch (error) {
      setResult({
        transcription: audioFiles[selectedAudio].transcription,
        output: `Error: ${error.message}`,
      });
      setStatus(500);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <Accordion
      title={labels.title}
      content={
        <div id="runCommand">
          <span className="description">{labels.full}</span>

          {!audioFiles || selectedAudio === null ? (
            <span className="warning">⚠️ Warning: You must select an audio file.</span>
          ) : !isAudioValid ? (
            <span className="warning">
              ⚠️ Warning: The selected audio must have a full transcription to be sent to the model.
            </span>
          ) : null}

          <button
            className={`execute ${isLoading ? "disabled" : ""}`}
            onClick={handleAnalyze}
            disabled={isLoading || !isAudioValid}
          >
            {isLoading ? "Analyzing..." : labels.button}
          </button>

          {result && (
            <>
              <span className="resultTitle">Status: {status}</span>
              <div ref={resultRef} className={`result ${status === 200 ? "success" : "fail"}`}>
                <div className="result-item">
                  <strong>Transcription:</strong>
                  <div className="transcription">{result.transcription}</div>
                  
                  <strong>Server Output:</strong>
                  <div className="output">{result.output}</div>
                </div>
              </div>
            </>
          )}
        </div>
      }
    />
  );
};

export default EmotionRecognition;