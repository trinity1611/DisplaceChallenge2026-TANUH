# DISPLACE MedAI - Audio to ASR and Summarization Microservice

## 📖 Overview
The **DISPLACE MedAI Microservice** is a full-stack, GPU-accelerated application designed to transcribe and summarize medical audio recordings. It takes raw `.wav` audio files as input, identifies different speakers (Diarization), transcribes the conversation (ASR), and leverages Large Language Models (LLMs) to generate structured clinical summaries.

This tool is built specifically to handle the heavy computational loads of modern AI models while maintaining a lightweight footprint, achieved via state-of-the-art **4-bit NF4 quantization**.

---

## ✨ Key Features
- **Speaker Diarization:** Accurately distinguishes between different speakers (e.g., Doctor vs. Patient) using `pyannote.audio`.
- **Automatic Speech Recognition (ASR):** Highly accurate multilingual transcription powered by OpenAI's Whisper model.
- **Clinical Summarization:** Extracts critical medical context and outputs structured summaries using LLMs (e.g., Qwen / LLaMA).
- **VRAM Optimized:** Implements `bitsandbytes` 4-bit NormalFloat (NF4) double quantization, allowing massive 3B+ parameter models to run smoothly on consumer GPUs (8GB VRAM).
- **Real-Time Polling UI:** Beautiful, modern glassmorphism frontend that tracks pipeline progress in real-time.
- **Privacy-First Design:** Fully non-persistent. Audio files and summaries are processed in memory and purged entirely once the session ends. No sensitive patient data is permanently stored.

---

## 🏗️ Architecture

### Frontend
- **Tech Stack:** Vanilla HTML5, CSS3, JavaScript.
- **Design:** Modern grid-based layout, responsive design, interactive SVG iconography, and real-time asynchronous status polling.
- **Server:** Served statically via the FastAPI backend.

### Backend & AI Pipeline
- **API Framework:** FastAPI running on Uvicorn.
- **Database:** SQLite (used temporarily to track ephemeral background job statuses).
- **Pipeline Components:**
  1. **Audio Validation:** Checks format (16kHz `.wav`) and size constraints.
  2. **Diarization Engine:** `pyannote/speaker-diarization-3.1`.
  3. **ASR Engine:** `openai/whisper-small` (or variants).
  4. **Summarization Engine:** Transformer-based causal language models (e.g., `Qwen/Qwen2.5-3B-Instruct`), heavily optimized with 4-bit quantization.

---

## 💻 Hardware Requirements
To run the AI pipeline locally without Out-Of-Memory (OOM) crashes, the following is recommended:
- **OS:** Windows / Linux
- **GPU:** NVIDIA GPU with at least **8GB VRAM** (e.g., RTX 3060, 4060, or better).
- **RAM:** 16GB+ System RAM.
- **CUDA:** NVIDIA CUDA Toolkit installed and configured.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
Ensure you are in the `microservice` directory of the project.
```bash
cd DISPLACE-2026-Baselines/microservice
```

### 2. Set Up the Conda Environment
It is highly recommended to use Anaconda or Miniconda.
```bash
conda create -n medai python=3.10
conda activate medai
```

### 3. Install Dependencies
Install PyTorch with CUDA support first, then install the remaining requirements.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U bitsandbytes>=0.46.1 accelerate
```
*(Note: `bitsandbytes` and `accelerate` are strictly required for 4-bit model quantization on Windows/Linux).*

### 4. Hugging Face Authentication
The pipeline uses gated models (like PyAnnote). You must authenticate with Hugging Face.
1. Create a `.env` file in the `microservice` directory.
2. Add your Hugging Face Access Token:
```env
HF_TOKEN=your_huggingface_token_here
```
*(Ensure you have accepted the user agreements for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face).*

---

## 🏃‍♂️ Running the Service

Start the backend server using Uvicorn:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
Once the server is running, open your web browser and navigate to:
**👉 `http://localhost:8000`**

---

## 📖 Usage Guide

1. **Prepare Audio:** Ensure your audio is a `.wav` file, ideally sampled at 16kHz, and is under 25 MB in size.
2. **Upload:** Click the "Choose Audio file..." box on the web interface to select your recording.
3. **Select Language:** Choose the primary language (Hindi or Kannada) from the dropdown.
4. **Process:** Click **Process**. The backend will sequentially run Diarization -> ASR -> Summarization. This typically takes **7 to 10 minutes** depending on your GPU. *Do not refresh the tab during this time.*
5. **Export:** Once complete, the output window will display the Clinical Summary and Transcript. Click the **Download Results** button to save a `.txt` file of the insights directly to your computer.

---

## 🔌 API Reference

The backend exposes several RESTful endpoints:

- `POST /upload`
  - **Body:** `multipart/form-data` containing the `file` and `language`.
  - **Response:** Returns a `job_id`.
- `GET /status/{job_id}`
  - **Response:** Returns the current progress, status (`processing`, `completed`, `failed`), and any error messages.
- `GET /results/{job_id}`
  - **Response:** Returns the final JSON payload containing the transcript and clinical summary.

---

## 🔒 Privacy & Security
This application is strictly non-persistent. 
- All `.wav` files temporarily saved to the `uploads/` directory are purged upon job completion or failure.
- In-memory results are discarded. 
- The SQLite database only tracks transient `job_id` statuses and does not log transcript text or PII (Personally Identifiable Information).

---
*Built for the DISPLACE 2026 Challenge.*
