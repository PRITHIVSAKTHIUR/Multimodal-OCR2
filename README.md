# **Multimodal-OCR2**

Multimodal-OCR2 is an advanced, experimental optical character recognition and document analysis suite designed to extract high-fidelity text, reconstruct complex document layouts, and generate structured markdown from diverse visual inputs. Built around a versatile collection of state-of-the-art vision-language models—including architectures based on Qwen2.5-VL, Qwen3-VL, and specialized document parsers like SmolDocling—this application excels at handling dense documents, multilingual texts, and real-world scene text. The suite features a custom-built, interactive web interface that allows users to seamlessly process standard documents, receipts, and screenshots. With built-in support for advanced parsing techniques (such as Docling integration for structured markdown export) and fully GPU-accelerated inference, Multimodal-OCR2 provides developers and researchers with a powerful environment for testing and refining next-generation document intelligence workflows.

<img width="1920" height="1797" alt="Screenshot 2026-03-22 at 12-29-30 Multimodal OCR2 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/386f9d7f-7377-44d5-b9c9-514c0640564f" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized vision-language models directly from the interface. Supported models include `FireRed-OCR`, `Nanonets-OCR-s`, `MonkeyOCR-Recognition`, `Thyme-RL`, `Typhoon-OCR-7B`, and `SmolDocling-256M-preview`.
* **Advanced Document Parsing:** Specialized integration with SmolDocling allows for deep document understanding, translating visual elements like charts, code blocks, and tables directly into structured Markdown output.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a `.txt` file.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run Multimodal-OCR2 locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`. Note that this suite also requires `docling_core` for advanced markdown export.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
docling-core
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR2.git](https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR2.git)
