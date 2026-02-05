  import React, { useState, useEffect, useRef } from "react";
  import Accordion from "../accordion";
  import TextInput from "../textInput";

  const AudioSelection = ({ audioFiles, setAudioFiles, selectedAudio, setSelectedAudio }) => {
    const [isRecording, setIsRecording] = useState(false);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const startRecording = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      try {
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];

        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
            console.log("Chunk received:", event.data.size);
          }
        };

        mediaRecorderRef.current.onstop = () => {
          console.log("Recorder stopped. Chunks count:", audioChunksRef.current.length);
          if (!audioChunksRef.current.length) {
            console.warn("No audio chunks available on stop");
            return;
          }

          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(audioBlob);
          audioChunksRef.current = [];

          const reader = new FileReader();
          reader.readAsDataURL(audioBlob);

          reader.onloadend = async () => {
            const base64 = reader.result.split(",")[1];
            console.log("Base64 audio generated", base64.length);

            let transcription = "";

            try {
              const response = await fetch("https://uncombustible-collinearly-corene.ngrok-free.dev/asr/transcribe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ encoded_audio: base64 }),
              });
              const result = await response.json();
              transcription = result.transcription || "";
            } catch (error) {
              console.error("Transcription failed:", error);
            } finally {
              setAudioFiles((prevFiles) => [
                ...prevFiles,
                {
                  ID: `audioFile_${Date.now()}`,
                  encoded_audio: base64,
                  audioUrl,
                  transcription,
                },
              ]);
            }
          };
        };

        mediaRecorderRef.current.start();
        console.log("Recording started");

        // Stop recording after 5 seconds (adjust as needed)
        setTimeout(() => {
          mediaRecorderRef.current.stop();
        }, 5000);

      } catch (error) {
        console.error("Error accessing microphone:", error);
      }
    };

    
    const stopRecording = () => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    };

    useEffect(() => {
      if (isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isRecording]);

    const updateAudioFileID = (index, newID) => {
      setAudioFiles((prevFiles) => {
        const updatedFiles = [...prevFiles];
        updatedFiles[index] = { ...updatedFiles[index], ID: newID };
        return updatedFiles;
      });
    };

    const updateAudioFileTranscription = (index, newTranscription) => {
      setAudioFiles((prevFiles) => {
        const updatedFiles = [...prevFiles];
        updatedFiles[index] = { ...updatedFiles[index], transcription: newTranscription };
        return updatedFiles;
      });
    };

    const deleteAudioFile = (index) => {
      setAudioFiles((prevFiles) => prevFiles.filter((_, i) => i !== index));
    };

    const sendSelectAudio = (audioData, index) => {
      if (audioData) {
        setSelectedAudio(index);
      }
    };

    const handleFileUpload = async (event) => {
      const file = event.target.files[0];

      if (file) {
        const fileReader = new FileReader();

        fileReader.onloadend = async () => {
          const fileContent = fileReader.result;

          if (file.name.endsWith(".json")) {
            try {
              const jsonData = JSON.parse(fileContent);
              setAudioFiles((prevFiles) => [
                ...prevFiles,
                {
                  ID: jsonData.ID,
                  encoded_audio: jsonData.encoded_audio,
                  audioUrl: `data:audio/wav;base64,${jsonData.encoded_audio}`,
                  transcription: jsonData.transcription || "", 
                },
              ]);
            } catch (error) {
              console.error("Error parsing JSON:", error);
            }
          } else if (file.name.endsWith(".wav")) {
            const arrayBuffer = await file.arrayBuffer();
            const audioBlob = new Blob([arrayBuffer], { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(audioBlob);

            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);

            reader.onloadend = async () => {
              const base64 = reader.result.split(",")[1];
              const fileNameWithoutExtension = file.name.replace(/\.[^/.]+$/, "");

              let transcription = "";
              try {
                const response = await fetch("https://uncombustible-collinearly-corene.ngrok-free.dev/asr/transcribe", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ encoded_audio: base64 }),
                });
                const result = await response.json();
                transcription = result.transcription || "";
              } catch (error) {
                console.error("Transcription failed for .wav upload:", error);
              }

              setAudioFiles((prevFiles) => [
                ...prevFiles,
                {
                  ID: fileNameWithoutExtension,
                  encoded_audio: base64,
                  audioUrl: audioUrl,
                  transcription,
                },
              ]);
            };
          }
          else {
            alert("Please upload a valid .wav or .json file.");
          }
        };

        fileReader.readAsText(file);
      }
    };

    const downloadWavFile = (audioData) => {
      const audioBlob = new Blob([new Uint8Array(atob(audioData.encoded_audio).split("").map(c => c.charCodeAt(0)))], { type: "audio/wav" });
      const url = URL.createObjectURL(audioBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${audioData.ID}.wav`;
      link.click();
      URL.revokeObjectURL(url);
    };

    const downloadJsonFile = (audioData) => {
      const jsonData = JSON.stringify(audioData);
      const blob = new Blob([jsonData], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${audioData.ID}.json`;
      link.click();
      URL.revokeObjectURL(url);
    };

    return (
      <Accordion
        title="Audio selection"
        content={
          <div className="audio-list">
            <span className="description">
            Allows users to record audio, select preloaded audio files, upload new files in .wav or .json format, and send them to a backend for later processing. It also allows users to download the files in .wav or .json format.
            </span>

            {(audioFiles.length && audioFiles.map((file, index) => (
              <div key={file.ID} className="audio-row">
                <span className="number">{index + 1}.</span>
                <TextInput
                  value={file.ID}
                  setValue={(newID) => updateAudioFileID(index, newID)}
                  placeholder="Audio ID"
                  hasError={file.ID === ""}
                />
                <audio controls src={file.audioUrl}></audio>
                <button
                  onClick={() => sendSelectAudio(file, index)}
                  className={`sendAudioButton ${selectedAudio === index ? "disabled" : ""}`}
                >
                  {selectedAudio === index ? "Audio selected" : "Select the audio"}
                </button>
                <div className="vertical one"><div></div></div>
                <button className="download" title="Download JSON file" onClick={() => downloadJsonFile(file)}>
                  <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="icon">
                    <polyline points="8 12 12 16 16 12" fill="none" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" />
                    <path d="M5 21H19M12 3V16" fill="none" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" />
                  </svg>
                  <span>.json</span>
                </button>
                <button className="download" title="Download .wav file" onClick={() => downloadWavFile(file)}>
                  <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="icon">
                    <polyline points="8 12 12 16 16 12" fill="none" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" />
                    <path d="M5 21H19M12 3V16" fill="none" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" />
                  </svg>
                  <span>.wav</span>
                </button>
                <TextInput
                  className={"transcript"}
                  value={file.transcription}
                  setValue={(newTranscription) => updateAudioFileTranscription(index, newTranscription)}
                  placeholder="Transcription"
                />
                <div className="vertical two"><div></div></div>
                <button onClick={() => deleteAudioFile(index)} className="deleteButton">
                  <svg width="25px" height="25px" viewBox="0 0 32 32" id="Outlined" xmlns="http://www.w3.org/2000/svg">
                    <g id="Fill">
                      <rect height="12" width="2" x="15" y="12"/>
                      <rect height="12" width="2" x="19" y="12"/>
                      <rect height="12" width="2" x="11" y="12"/>
                      <path d="M20,6V5a3,3,0,0,0-3-3H15a3,3,0,0,0-3,3V6H4V8H6V27a3,3,0,0,0,3,3H23a3,3,0,0,0,3-3V8h2V6ZM14,5a1,1,0,0,1,1-1h2a1,1,0,0,1,1,1V6H14ZM24,27a1,1,0,0,1-1,1H9a1,1,0,0,1-1-1V8H24Z"/>
                    </g>
                  </svg>
                </button>
              </div>
            ))) || <span className="noAudio">No audio Created</span>}

            <div className="separation"></div>
            <div className="horizontal">
              <button className={`recordingButton ${isRecording ? "recording" : ""}`} onClick={() => setIsRecording(!isRecording)}>
                {isRecording ? "Stop Recording" : "Start Recording"}
              </button>
              <label className="uploadButton">
                <span>Upload a json/wav file</span>
                <input
                  type="file"
                  accept=".wav,.json"
                  onChange={handleFileUpload}
                  className="upload"
                />
              </label>
            </div>
          </div>
        }
      />
    );
  };

  export default AudioSelection;