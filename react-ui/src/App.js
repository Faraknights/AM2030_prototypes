import './App.css';
import AudioSelection from './components/paths/audioSelection';
import EmotionRecognition from './components/paths/emotionRecognition';
import { useEffect, useState } from 'react';

function App() {
  const [audioFiles, setAudioFiles] = useState([]);
  const [selectedAudio, setSelectedAudio] = useState(null);
  
  useEffect(() => {
    const fetchAudioFiles = async () => {
      try {
        const fileList = [
          "Beatles.json",
          "Dumb.json",
          "Rock.json"
        ];

        const fileDataPromises = fileList.map(async (fileName) => {
          const fileResponse = await fetch(`/audioSamples/${fileName}`);
          const fileData = await fileResponse.json();

          return {
            ID: fileData.ID,
            encoded_audio: fileData.encoded_audio,
            audioUrl: `data:audio/wav;base64,${fileData.encoded_audio}`,
            dialogue: fileData.dialogue || "F",
            transcription: fileData.transcription || "",
          };
        });

        const files = await Promise.all(fileDataPromises);
        setAudioFiles(files);
      } catch (error) {
        console.error("Error fetching audio files:", error);
      }
    };

    fetchAudioFiles();
  }, []);

  return (
    <div className="App">
      <div className='main'>
        <h1>~ 𝓟𝓻𝓸𝓽𝓸𝓽𝔂𝓹𝓮 ~</h1>
        <AudioSelection
          audioFiles={audioFiles}
          setAudioFiles={setAudioFiles}
          selectedAudio={selectedAudio}
          setSelectedAudio={setSelectedAudio}
        />
        <EmotionRecognition
          audioFiles={audioFiles}
          selectedAudio={selectedAudio}
        />
      </div>
    </div>
  );
}

export default App;
