# Clinical Report Simplifier 🩺

An AI-powered assistant designed to bridge the communication gap between healthcare providers and patients by translating complex medical jargon into simple, actionable information.

## 📌 Project Overview
Medical reports are often filled with terminology that can be confusing or stressful for patients. This tool uses **Large Language Models (GPT-4o)** to:
* **Simplify:** Translate medical jargon into plain, everyday language.
* **Summarize:** Provide a clear overview of clinical findings.
* **Structure:** Organize tests and measurements into easy-to-read tables.
* **Empower:** Generate a list of relevant questions for patients to ask their doctors.

## 🛠️ Technical Workflow
The application uses a **Map-Reduce** architecture to handle documents of any length:
1. **Extraction:** The PDF is parsed using `pypdf`.
2. **Chunking:** The text is split into segments to stay within the model's context window.
3. **Mapping:** Each segment is analyzed to extract findings and definitions using **OpenAI Structured Outputs**.
4. **Reducing:** All extractions are merged into one final, coherent report.
5. **Safety Review:** A final AI pass ensures the tone is helpful and not alarmist.

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.9 or higher.
* An OpenAI API Key.

### 2. Installation
Install the necessary dependencies via terminal:
```bash
pip install requirements.txt
```

### 3. Running the App
Launch the Streamlit server:

```bash
streamlit run app.py
```

### 4. How to Use
Open the local URL (usually http://localhost:8501).

Input your OpenAI API Key in the sidebar.

Upload a PDF clinical report.

Click "Lancer l'analyse" and navigate through the generated tabs.

### Note 
It can take some time to process the file.
