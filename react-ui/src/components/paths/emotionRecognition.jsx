import { useEffect, useRef, useState } from "react";
import Accordion from "../accordion";

const labels = {
  title: "Emotion and Intention Recognition",
  full:
    "This module will recognize both the emotion and the intention in the audio sent through the module [Audio Selection]. It will first segment if needed, then transcribe the text before analyzing the voice and the text to determine the user's emotional state and explicit intentions.",
  buttonEmotion: "Emotion",
  buttonIntention: "Intention",
  buttonBoth: "Both",
};

// Unified API call function
const callASRApi = async (endpoint, bodyObj) => {
  const response = await fetch(`http://localhost:5000/asr/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj),
  });
  const data = await response.json();
  return {
    status: response.status,
    result: data.result || data.error || "No result received",
  };
};

function extractIntentName(predictedText, validIntents, distanceThreshold = 5, fallback = "no_intention") {
    function getLastTokensFromLines(text) {
        return text
            .split('\n')
            .map((line, index) => {
                const tokens = line.toLowerCase().trim().match(/\b\w+\b/g);
                return {
                    index,
                    line,
                    lastToken: tokens && tokens.length > 0 ? tokens[tokens.length - 1] : null
                };
            })
            .filter(entry => entry.lastToken !== null);
    }

    function levenshtein(a, b) {
        const matrix = Array.from({ length: b.length + 1 }, (_, i) =>
            Array.from({ length: a.length + 1 }, (_, j) => 0)
        );

        for (let i = 0; i <= b.length; i++) matrix[i][0] = i;
        for (let j = 0; j <= a.length; j++) matrix[0][j] = j;

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b[i - 1] === a[j - 1]) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + 1
                    );
                }
            }
        }

        return matrix[b.length][a.length];
    }

    const candidates = getLastTokensFromLines(predictedText);
    const lines = predictedText.split('\n');

    let bestMatch = null;
    let bestLineIndex = -1;
    let minDistance = distanceThreshold;

    for (const { lastToken, index } of candidates) {
        for (const intent of validIntents) {
            const intentLower = intent.toLowerCase();
            const distance = levenshtein(lastToken, intentLower);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = intent;
                bestLineIndex = index;
            }
        }
    }

    const chainOfThought = lines.filter((_, i) => i !== bestLineIndex);

    return {
        chainOfThought,
        code: bestMatch || fallback
    };
}


// Unified response processor
export const processResponse = (rawResult) => {
  const lines = rawResult.trim().split("\n");
  const lastLine = lines.pop();
  let code = [];

  try {
    code = JSON.parse(lastLine);
  } catch {
    code = lastLine
    // eslint-disable-next-line
      .replace(/[\[\]"'*]/g, "")
      .split(",")
      .map((e) => e.trim())
      .filter(Boolean);
  }

  const chainOfThought = lines.join("\n").trim();

  return { chainOfThought, code };
};

const EmotionRecognition = ({ audioFiles, selectedAudio }) => {
  const [responseMessage, setResponseMessage] = useState([]);
  const [status, setStatus] = useState(null);
  const [loadingMode, setLoadingMode] = useState(null);
  const resultRef = useRef(null);
  const [intents, setIntents] = useState([]);
  const [categories, setCategories] = useState([]);
  const [intentsData, setIntentsData] = useState({})

  useEffect(() => {
    fetch('/assets/intents.json')
      .then(response => response.json())
      .then(data => {
        setIntents([...data.map(intent => intent.intent), "no_intention"]);

        // Store descriptions and parameters for each intention
        const mappedData = data.reduce((acc, intent) => {
          acc[intent.intent] = {
            description: intent.description || "No description available",
            parameters: intent.parameters || []
          };
          return acc;
        }, {});
        setIntentsData(mappedData);
      });

    fetch('/assets/categories.json')
      .then(response => response.json())
      .then(data => setCategories(data.map(category => category.name)));
  }, []);

  useEffect(() => {
    if (resultRef.current) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [responseMessage]);

  const isAudioValid =
    audioFiles &&
    selectedAudio !== null &&
    audioFiles[selectedAudio] &&
    audioFiles[selectedAudio].transcription &&
    audioFiles[selectedAudio].transcription.trim() !== "";

  const processSentence = async (sentence, mode) => {
    console.log("Processing sentence:", sentence, "Mode:", mode);
    setResponseMessage((prev) => [
      ...prev,
      {
        text: sentence,
        emotionCoT: "",
        intentionCoT: "",
        fineIntentionCoT: "",
        categoryCode: "",
        emotions: [],
        intention: "",
        parameters: null, // NEW
        status: null,
      },
    ]);

    if (mode === "emotion" || mode === "both") {
      const { status: emotionStatus, result: emotionRaw } = await callASRApi("emotion", { transcription: sentence });
      const { chainOfThought: emotionCoT, code: emotions } = processResponse(emotionRaw);

      setResponseMessage((prev) =>
        prev.map((item) =>
          item.text === sentence
            ? { ...item, emotionCoT, emotions, status: emotionStatus }
            : item
        )
      );
    }

    if (mode === "intention" || mode === "both") {
      const { result: intentionRaw } = await callASRApi("intention_category", { transcription: sentence });
      const { chainOfThought: intentionCoT, code: categoryCode } = extractIntentName(intentionRaw, categories, 1, "NONE");

      setResponseMessage((prev) =>
        prev.map((item) =>
          item.text === sentence
            ? { ...item, intentionCoT, categoryCode }
            : item
        )
      );

      if (categories.includes(categoryCode)) {
        const { status: fineStatus, result: fineIntentionRaw } = await callASRApi("intention", {
          category_code: categoryCode,
          transcription: sentence,
        });

        const { chainOfThought: fineIntentionCoT, code: intention } = extractIntentName(fineIntentionRaw, intents, 5, "no_intention");

        setResponseMessage((prev) =>
          prev.map((item) =>
            item.text === sentence
              ? { ...item, fineIntentionCoT, intention }
              : item
          )
        );

        setStatus(fineStatus);

        if (intention && intention !== "no_intention") {
          const currentIntent = intentsData[intention] || {};
          const intentionDescription = currentIntent.description || "No description available";
          const intentionParams = currentIntent.parameters || [];

          const parametersPayload = {
            category_name: categoryCode,
            category_description: "To be provided if available", // you can also fetch from categories.json if it contains descriptions
            intention_name: intention,
            intention_description: intentionDescription,
            parameters_list: intentionParams.map(param => ({
              name: param.name,
              description: param.description
            })),
            user_utterance: sentence,
          };

          const { result: parametersRaw } = await callASRApi("parameters", parametersPayload);

          let parsedParameters;
          try {
            parsedParameters = JSON.parse(parametersRaw);
          } catch {
            parsedParameters = parametersRaw;
          }

          setResponseMessage(prev =>
            prev.map(item =>
              item.text === sentence
                ? { ...item, parameters: parsedParameters }
                : item
            )
          );
        }

      } else {
        setResponseMessage((prev) =>
          prev.map((item) =>
            item.text === sentence
              ? { ...item, fineIntentionCoT: "", intention: "no_intention" }
              : item
          )
        );
        setStatus(200);
      }
    }
  };


  const handleRunCommand = async (mode) => {
    setResponseMessage([]);
    setStatus(null);

    if (!isAudioValid) {
      setResponseMessage([{ text: "No audio transcription available.", output: "" }]);
      setStatus(400);
      return;
    }

    setLoadingMode(mode);  // Set loading mode here

    try {
      const { transcription, segment } = audioFiles[selectedAudio];

      if (segment === "T") {
        const sentences = transcription
          .split(/[.,!?]/)
          .map((s) => s.trim())
          .filter(Boolean);

        for (const sentence of sentences) {
          await processSentence(sentence, mode);
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      } else {
        await processSentence(transcription, mode);
      }
    } catch (error) {
      setResponseMessage([
        { text: "Error", chainOfThought: "", emotions: [], intention: "", output: error.message },
      ]);
      setStatus(500);
    } finally {
      setLoadingMode(null); // Reset loading mode when done
    }
  };

  return (
    <Accordion
      title={labels.title}
      content={
        <div id="runCommand">
          <span className="description">{labels.full}</span>

          {!audioFiles || selectedAudio === null ? (
            <span className="warning">⚠ Warning: You must select an audio file.</span>
          ) : !isAudioValid ? (
            <span className="warning">
              ⚠ Warning: The selected audio must have a full transcription to be sent to the model.
            </span>
          ) : null}

          <div style={{ display: "flex" }}>
            <button
              className={`execute ${loadingMode === "emotion" ? "disabled" : ""} ${loadingMode && loadingMode !== "emotion" ? "otherRun" : ""}`}
              onClick={() => handleRunCommand("emotion")}
              disabled={loadingMode === "emotion" || !isAudioValid}
              style={{ marginRight: "0.5rem" }}
            >
              {loadingMode === "emotion" ? "Running..." : labels.buttonEmotion}
            </button>

            <button
              className={`execute ${loadingMode === "intention" ? "disabled" : ""} ${loadingMode && loadingMode !== "intention" ? "otherRun" : ""}`}
              onClick={() => handleRunCommand("intention")}
              disabled={loadingMode === "intention" || !isAudioValid}
              style={{ marginRight: "0.5rem" }}
            >
              {loadingMode === "intention" ? "Running..." : labels.buttonIntention}
            </button>

            <button
              className={`execute ${loadingMode === "both" ? "disabled" : ""} ${loadingMode && loadingMode !== "both" ? "otherRun" : ""}`}
              onClick={() => handleRunCommand("both")}
              disabled={loadingMode === "both" || !isAudioValid}
            >
              {loadingMode === "both" ? "Running..." : labels.buttonBoth}
            </button>
          </div>

          {responseMessage.length > 0 && (
            <>
              <span className="resultTitle">Status: {status}</span>
              <div ref={resultRef} className={`result ${status === 200 ? "success" : "fail"}`}>
                {responseMessage.map(({ text, emotionCoT, intentionCoT, fineIntentionCoT, categoryCode, emotions, intention, parameters }, index) => (
                  <div key={index} className="result-item" style={{ marginBottom: "1rem" }}>
                    <strong>Input:</strong> <span className="input-text">{text}</span>

                    {emotionCoT && (
                      <div className="chain-of-thought emotion-cot">
                        {emotionCoT}
                      </div>
                    )}

                    {emotions && emotions.length > 0 && (
                      <div className="code">
                        {emotions.join(", ")}
                      </div>
                    )}

                    {intentionCoT && (
                      <div className="chain-of-thought intention-category-cot">
                        {intentionCoT}
                      </div>
                    )}

                    {categoryCode && (
                      <div className="code">
                        {Array.isArray(categoryCode) ? categoryCode.join(", ") : categoryCode}
                      </div>
                    )}

                    {fineIntentionCoT && (
                      <div className="chain-of-thought fine-intention-cot">
                        {fineIntentionCoT}
                      </div>
                    )}

                    {intention && (
                      <div className="code">
                        {intention}
                      </div>
                    )}

                    {parameters && (
                      <div className="code">
                        {parameters}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      }
    />
  );
};
export default EmotionRecognition;
