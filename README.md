This repository implements the QLoRA adapter of **Llama-3-8B-Instruct**, fine-tuned to translate natural language into **First-Order Logic (FOL)** and **Linear Temporal Logic (LTL)**.  
It is specifically adapted to capture **user preferences and likes** expressed in natural language utterances.

# Installation with Docker

You can deploy the project using either CPU or GPU, depending on your system.

### CPU Deployment

Run the following command:

```
docker-compose -f docker-compose_cpu.yml up --build
```

### GPU Deployment

If you have an NVIDIA GPU and CUDA installed:

```
docker-compose -f docker-compose_gpu.yml up --build
```

# Interfaces

- Backend Swagger API: http://localhost:5000/apidocs  
- React frontend UI: http://localhost:3000

# API Endpoints

1. **Check GPU availability**

   GET /gpu  
   Returns JSON indicating whether a GPU is available for processing.

   Example response:
   ```
   {
     "gpu_available": true
   }
   ```

2. **Generate FOL/TTL from transcription**

   POST /asr/folltl  
   Accepts a JSON body with a transcription and returns the generated FOL/TTL text or an error.

   Example request:
   ```
   {
     "text": "I like the song Dumb from Nirvana."
   }
   ```

   Example response:
   ```
   {
     "generated_text": "(likes I (song Dumb Nirvana))"
   }
   ```

# Notes

- Use the CPU or GPU Docker Compose command depending on your environment.  
- Both Swagger and the React UI are accessible simultaneously; the React UI provides a more user-friendly interface for testing.  
- The backend API can be accessed directly for integration or scripting purposes.
