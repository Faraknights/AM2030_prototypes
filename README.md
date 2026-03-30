# AM2030 Prototypes

This repository contains prototypes developed within the **AM2030 project**, focusing on intelligent in-vehicle assistants designed to operate in **resource-constrained environments** such as embedded automotive systems.

## Overview

The goal of this project is to design a conversational interface capable of:

- Understanding user intentions
- Detecting emotional states
- Extracting structured knowledge from natural language

All components are built with **Small Language Models (SLMs)** to ensure compatibility with **limited computational environments** like cars.

---

## Key Features

### 1. Emotion Detection

The system performs **sentiment analysis** based on the **Ekman model**, detecting 7 core emotions:

- Joy  
- Sadness  
- Anger  
- Fear  
- Disgust  
- Surprise  
- Neutral  

This enables the assistant to adapt its responses according to the user’s emotional state.

---

### 2. Intent Detection Pipeline

Intent recognition is handled using a **multi-step LLM-based pipeline**, designed for robustness and scalability:

#### Step 1: Category Classification
- The user input is first classified into a **high-level category**.

#### Step 2: Intent Classification
- Within the predicted category, a **specific intent** is selected.
- The system supports over 100fine-grained intents.

#### Step 3: Parameter Extraction
- Relevant **entities and parameters** are extracted from the user query.

This decomposition approach improves accuracy and reduces ambiguity compared to single-step classification.

---

### 3. Preference Detection (Branch: `FOL_Bastien`)

An extended module is available in the `FOL_Bastien` branch.

#### Objective
Detect and formalize **user preferences** expressed in natural language.

The pipeline works such as follows:

1. Identify preferences in the user sentence
2. Decompose the sentence into relevant entities
3. Translate the extracted information into **First-Order Logic (FOL)**

#### example

![Environment](/src/6e023ce6-37a4-4123-9b20-70bf17fb31fd.png)

