# Project OASIS: Offline Adaptive Student & Instructor System


<img width="1456" height="720" alt="image" src="https://github.com/user-attachments/assets/fc857cbb-eb51-43ef-908d-d6a688cd14f0" />



![Gemma 4](https://img.shields.io/badge/Powered_by-Gemma_4-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge)
![Offline](https://img.shields.io/badge/Status-100%25_Offline-success?style=for-the-badge)

**OASIS** is an offline, Gemma 4-powered engineering tutor and interactive workspace, specifically designed to meet the unique requirements of both students and instructors in the Electronics and Communication Engineering (ECE) domain.

A Proof-of-Concept for modular, offline AI orchestration, demonstrating multi-stage data pipelines, agentic retrieval, and persistent state execution on edge hardware.

<p align="center">
  <img src="https://github.com/Mouni-kalamraju/Project_OASIS/blob/main/test_assets/Screenshot1.png" width="24%" />
  <img src="https://github.com/Mouni-kalamraju/Project_OASIS/blob/main/test_assets/Screenshot2.png" width="24%" />
  <img src="https://github.com/Mouni-kalamraju/Project_OASIS/blob/main/test_assets/Screenshot3.png" width="24%" />
  <img src="https://github.com/Mouni-kalamraju/Project_OASIS/blob/main/test_assets/Screenshot4.png" width="24%" />
</p>

---

## 🌟 The Vision

Engineering is the foundation of the future, yet the tools we use to teach it often leave students behind. While cloud-based AIs are powerful, they demand persistent, high-speed internet. Furthermore, the unique challenges of the ECE domain—such as interpreting complex circuit diagrams, processing signals, and analyzing block diagrams—mean standard text-based offline AIs (which rely on traditional OCR) often fail catastrophically. 

**OASIS** changes this. Operating entirely offline at the edge, it empowers students to gain instant lab experience alongside their studies. It serves as a unified workspace where students can learn theory, simulate signal processing, and solve complex GATE (Graduate Aptitude Test in Engineering) questions, all without requiring an internet connection or a massive GPU.

**My ultimate vision is that a fully tailored OASIS ecosystem becomes a standard, built-in tool on every engineering student's laptop—serving as the modern, interactive equivalent of a textbook.**

---

## 🏗️ Dual-Tiered Architecture

We solved the edge-hardware bottleneck by splitting the workload across the Gemma 4 family:

1. **For the Educator (Gemma 4 26B - Cloud/Server):** 
   Runs asynchronously on a local school server or cloud GPU. It visually ingests decades of complex GATE Exam PDFs. Bypassing traditional OCR, it uses programmatic vector-slicing to crop circuits, solves the advanced math autonomously, and outputs a perfectly structured, syllabus-categorized Vector Database (FAISS Knowledge Base).
2. **For the Student (Gemma 4 E4B - Edge Device):** 
   Runs completely offline on low-power student laptops. It acts as a hands-on interactive tutor and DSP coding sandbox, utilizing the pre-processed educator database to provide lightning-fast, highly accurate explanations natively on the CPU.

---

## 🛠️ Core Features

### 1. 🎛️ The DSP Workspace (Stateful Code Execution)
*   **Hands-on Learning:** Students upload raw signal data (`.csv`) and ask Gemma to apply filters (e.g., "Apply a 60Hz Low Pass Filter").
*   **Native Code Execution:** The `e4b` model natively generates Python code (`scipy`, `pandas`, `matplotlib`). The app executes this code in a **Stateful Python Sandbox**, rendering the plots locally.
*   **Dynamic UI Tuning:** The AI autonomously extracts tuning variables and generates interactive Streamlit sliders (e.g., Cutoff Frequencies) for the student to manipulate the math in real-time.

### 2. 🎓 The GATE Tutor (Agentic RAG & Multimodal Vision)
*   **Agentic Retrieval:** When a student uploads a circuit diagram and asks a question, the local `e4b` model triggers a FAISS Vector Search. It cross-references the student's prompt with historical data curated by the educator, alerting the student if a similar concept appeared in previous exams.
*   **No OCR Bottleneck:** We use PyMuPDF to extract text and dynamically crop embedded vector drawings from the exam papers, ensuring zero data loss on complex schematics.

---

## 🚀 How to Run Locally

OASIS is designed to run locally on consumer hardware. 

### Prerequisites
1. **Python 3.9+** installed on your system.
2. **Ollama** installed (Download from [ollama.com](https://ollama.com/)).
3. *Windows Users:* Ensure "Long Paths" are enabled in your registry if you encounter `pip` installation issues with Streamlit.

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Mouni-kalamraju/Project_OASIS.git
cd Project-OASIS
```
**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```
**3. Pull the required Gemma 4 models via Ollama:**
```bash
# Pull the edge model for the Student App
ollama pull batiai/gemma4-e4b:q4

# (Optional) Pull the 26B model if you intend to run the Educator Extraction scripts
ollama pull gemma4:26b
```

### Launch the App
If you are on Windows, simply double-click Start_Oasis.bat. Alternatively, run:
```bash
streamlit run student_app.py
```
(Tip: Disconnect your Wi-Fi after launching to experience the true offline capabilities!)

---
## 📂 Project Structure
*  **student_app.py**: The main Streamlit application (Student Edge Interface).
*  **build_db.py**: Educator script to compile the extracted JSON into a FAISS Vector Database.
*  **extract_images.py**: The programmatic PyMuPDF vector-slicing tool for circuit diagrams.
*  **gate_faiss_index/**: The local vector database containing historic exam solutions.
*  **requirements.txt**: Python dependencies.
---
## 🚧 Limitations & Future Roadmap
As a Proof of Concept, OASIS has laid the architectural groundwork, with a clear roadmap for future scaling:

*  **The Multimodal RAG Bottleneck**: Standard RAG pipelines rely on text. While our "Translation Pipeline" works well, future versions of OASIS will implement true Multimodal RAG using Vision Encoders (like SigLIP) to embed the raw circuit diagrams directly into the vector space, bypassing text entirely.
*  **Expanding the Sandboxes**: We fully implemented the Digital Signal Processing (DSP) Python sandbox. This modular architecture will eventually be expanded to include dedicated interactive UI sandboxes for Antenna Design (EMTL) and Control System Block Diagrams.
*  **E4B Vision Fine-Tuning**: To improve zero-shot circuit comprehension on edge devices, we plan to apply Parameter-Efficient Fine-Tuning (LoRA) to the E4B vision encoder using the synthetic datasets generated by the 26B educator model.
