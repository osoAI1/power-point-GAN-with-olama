🚀 AI Presentation Creator Pro
Local LLM-Powered Academic PowerPoint Generator with PDF-Enhanced RAG

AI Presentation Creator Pro is a fully local, offline, privacy-friendly tool that generates university-level academic PowerPoint presentations using a local GGUF LLM model (via llama.cpp).
The system optionally learns from any uploaded PDF, using TF-IDF retrieval (RAG) to produce accurate, domain-specific content for each slide.

This tool is ideal for lecturers, researchers, and students who want fast, high-quality academic slides directly from their own materials.

🏷️ Key Features
🔹 100% Local — No API Required

Works fully offline

Runs any .gguf LLaMA/Mistral/Qwen model

Powered by llama-cpp-python

🔹 PDF-Enhanced AI (RAG Mode)

Extracts text from PDF

Splits into meaningful chunks

Builds a TF-IDF similarity index

Uses relevant PDF content to guide slide content

Highly accurate academic writing

🔹 Smart Content Generator

2–3 polished paragraphs per slide

4–6 sentences per paragraph

Clean academic phrasing

Removes all “LLM-style” phrases

No bullet points, headings, or meta-comments

Two writing modes:

Prompt 1: Standard Academic

Prompt 2: Advanced Expert

🔹 Automatic Slide Splitting

Long content automatically generates extra slides.

🔹 Modern Web Interface

Built-in HTML/CSS/JS interface

Drag & drop PDF upload

Character counter

Animated progress modal

Error & success notifications

Professional UI/UX

🔹 Export to PowerPoint (.pptx)

Clean slide layout

Title slide

Slide headings

Wrapped paragraphs

Auto-generated additional slides for long content

🧠 Tech Stack
Component	Technology
Backend	Flask
AI Engine	llama-cpp-python
Model Format	GGUF
RAG	scikit-learn (TF-IDF + cosine similarity)
PDF Parsing	PyPDF2
PPTX Generator	python-pptx
Frontend	HTML, CSS, JS
📂 File Overview
presentation_creator_rag_pdf.py   # Full app (API + UI + RAG + PPT generator)
LICENSE                           # Apache 2.0 license
NOTICE                            # Additional attribution notice
qwen2.gguf                        # Local LLM model (user-provided)
static/                           # UI assets (inlined inside main file)
templates/                        # HTML templates (inlined in main file)

▶️ Getting Started
1. Install dependencies:
pip install flask python-pptx PyPDF2 scikit-learn llama-cpp-python

2. Add your GGUF model to project root:
qwen2.gguf

3. Start the app:
python presentation_creator_rag_pdf.py

4. Open in your browser:
http://127.0.0.1:5000

📸 Screenshots (Optional Section)

Add your screenshots here:

![UI Screenshot](images/ui.png)
![Slide Example](images/slide.png)

📜 License

This project is licensed under the Apache License 2.0.
Copyright (c) 2025 Osama Elemam

See the LICENSE file for details.

🙌 Author

Osama Elemam
AI/ML Engineer & Software Developer
Creator of AI Presentation Creator Pro

🎖️ Want to contribute?

Contributions, ideas, and feature requests are welcome!
Feel free to open a pull request or issue.
