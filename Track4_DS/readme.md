# The Third DISPLACE Challenge (2026) - DISPLACE-M
### Diarization and Speech Processing for Language understanding in Conversational Environments – Medical

[![Official Website](https://img.shields.io/badge/Official%20Website-DISPLACE%202026-blue?style=for-the-badge&logo=google-chrome)](https://displace2026.github.io/)

## 🏥 About
Inspired by the previous session of DISPLACE challenge, we have launched the **Third DISPLACE-M Challenge**. This challenge aims to advance diarization and speech understanding technologies in real-world healthcare conversations.

The dataset features medical conversations between Community Health Workers and local residents in **Hindi** and **Kannada**, collected across a wide geographic region. Key challenges include:
*   🗣️ Spontaneous dialogue & foreground speech overlap
*   🔊 Background speech & environmental noise
*   🌍 Dialectal variations in rural healthcare settings

## 📂 Track Information
This directory contains the official baseline implementation for:

### **Track 4 – Dialogue Summarization (DS)**
**Task:** Produce concise medical dialogue summaries.

---

## 1. Repository Structure
```text
Track4_DS/
├── data/               # Dataset directory 
├── configs/            # Configuration files for evaluation
├── scripts/            # Evaluation scripts
├── src/                # Core Python source code
├── utils/              # Helper functions and shared utilities
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 2. Baseline Setup

### Prerequisites
| Requirement | Description |
| :--- | :--- |
| **Dataset** | Download `Track_4_DS_DevData_1` from the [Official Website](https://displace2026.github.io/) and place it in the `data/` folder. |
| **Environment** | Ensure you have a working Python environment (Anaconda recommended). |

### Model Details
| Feature | Details |
| :--- | :--- |
| **Model Name** | `Llama-3.2-3B-Instruct` |
| **Repository** | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| **Parameters** | 3B |
| **Reference** | [Dubey et al., 2024](https://arxiv.org/abs/2407.21783) |

### Model Setup
| Step | Action |
| :--- | :--- |
| **1. Request Access** | Go to the [Hugging Face model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and request access. |
| **2. Generate Token** | Generate a **Hugging Face Token** from your account settings. |
| **3. Configure** | Update the `hf_token` field in `configs/summarization.yaml` with your token. |

### Installation
```bash
# Clone the repository
git clone https://github.com/displace2026/DISPLACE-2026-Baselines.git
cd DISPLACE-2026-Baselines/Track4_DS

# Install dependencies
pip install -r requirements.txt
```

## 3. Inference 🚀

We provide a unified script to run the complete pipeline, which includes **Automatic Speech Recognition (ASR)** followed by **Summarization**.

### Execution
Run the following command to submit the job:
```bash
bash run.sh
```

**Pipeline Flow:**
1.  **ASR**: Executes `../Track2_ASR/scripts/asr.py` (requires `Track2_ASR` setup).
2.  **Summarization**: Executes `scripts/summarization.py`.

**Output:**
*   Summarized files are saved in: `outputs/Summarization/`

## 4. Evaluation Metrics 📊

The primary metrics for **Track 4 – Dialogue Summarization** are:
*   **ROUGE-L**: Measures the longest common subsequence between the reference and generated summary, capturing sentence-level structure.
*   **BERTScore**: Computes similarity using contextual embeddings (BERT), focusing on semantic meaning rather than exact word overlap.

Metrics are saved to: `outputs/metrics/`

## 5. Baseline Results
Baseline results on the development set:

| Metric | Value |
| :--- | :--- |
| **ROUGE-L** | TBD |
| **BERTScore** | TBD |

> **Note:** These results serve as a reference. Participants are encouraged to improve upon them through model adaptation and decoding strategies.