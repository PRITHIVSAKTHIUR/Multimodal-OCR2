# **Multimodal-OCR2**

A comprehensive multimodal OCR application that supports both image and video document processing using state-of-the-art vision-language models. This application provides an intuitive Gradio interface for extracting text, converting documents to markdown, and performing advanced document analysis.

> [!note]
Demo here : https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR2

## Features

- **Multiple Model Support**: Choose from 4 different OCR models optimized for various use cases
- **Image Processing**: Extract text and convert documents from images
- **Video Processing**: Process video content with OCR capabilities
- **Document Conversion**: Convert documents to structured markdown format
- **Real-time Streaming**: Get results as they are generated
- **Advanced Configuration**: Fine-tune generation parameters for optimal results

## Supported Models

### SmolDocling-256M-preview
A multimodal Image-Text-to-Text model designed for efficient document conversion. Retains Docling's most popular features while ensuring full compatibility with Docling through seamless support for DoclingDocuments.

### Nanonets-OCR-s
A powerful, state-of-the-art image-to-markdown OCR model that goes far beyond traditional text extraction. It transforms documents into structured markdown with intelligent content recognition and semantic tagging.

### MonkeyOCR-Recognition
Adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.

### Typhoon-OCR-7B
A bilingual document parsing model built specifically for real-world documents in Thai and English. Extracts and interprets embedded text including chart labels and captions in both languages.


## Image / Video Inference Demo

![Screenshot 2025-06-20 at 12-51-03 OCR2 - a Hugging Face Space by prithivMLmods](https://github.com/user-attachments/assets/1448dc30-0f9c-4635-900b-b49d4baf3971)

---

![Screenshot 2025-06-20 at 12-49-32 OCR2 - a Hugging Face Space by prithivMLmods](https://github.com/user-attachments/assets/5e2d066c-7e38-4b3f-8bc2-d0cfdbc1e682)

---
![Screenshot 2025-06-20 at 12-52-08 OCR2 - a Hugging Face Space by prithivMLmods](https://github.com/user-attachments/assets/404899f4-14e0-4027-aaac-4c54fdeb3f2a)

---

https://github.com/user-attachments/assets/1df2349d-ecf1-43bc-adbf-3e4f9c3cc708

---

https://github.com/user-attachments/assets/f127606c-4d0b-468e-b7be-d124d2f9ee2b

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR2.git
cd Multimodal-OCR2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- torch
- transformers
- gradio
- spaces
- numpy
- PIL (Pillow)
- opencv-python
- docling-core

## Usage

### Running the Application

```bash
python app.py
```

The application will launch a Gradio interface accessible through your web browser.

### Image Processing

1. Select the "Image Inference" tab
2. Enter your query (e.g., "OCR the image", "Convert this page to docling")
3. Upload an image file
4. Choose your preferred model
5. Adjust advanced parameters if needed
6. Click Submit to process

### Video Processing

1. Select the "Video Inference" tab
2. Enter your query (e.g., "Explain the video in detail")
3. Upload a video file
4. Choose your preferred model
5. Adjust advanced parameters if needed
6. Click Submit to process

### Advanced Configuration

The application provides several tunable parameters:

- **Max New Tokens**: Maximum number of tokens to generate (1-2048)
- **Temperature**: Controls randomness in generation (0.1-4.0)
- **Top-p**: Nucleus sampling parameter (0.05-1.0)
- **Top-k**: Top-k sampling parameter (1-1000)
- **Repetition Penalty**: Penalty for repetitive text (1.0-2.0)

## Example Queries

### Image Processing
- "OCR the image"
- "Convert this page to docling"
- "Convert chart to OTSL"
- "Convert code to text"
- "Convert this table to OTSL"
- "Convert formula to latex"

### Video Processing
- "Explain the video in detail"
- "Extract text from video frames"

## Technical Details

### Model Loading
The application loads all models at startup using GPU acceleration when available. Models are loaded with 16-bit precision for optimal performance.

### Video Processing
Videos are processed by extracting 10 evenly spaced frames, which are then processed as a sequence of images by the selected model.

### SmolDocling-256M Special Features
- Automatic padding for OTSL and code conversion tasks
- Value normalization for OCR and element identification
- Advanced postprocessing for structured document output
- Automatic conversion to markdown format

### GPU Support
The application uses CUDA acceleration when available and falls back to CPU processing otherwise.

## Hardware Requirements

- **Minimum**: 12GB RAM, CPU
- **Recommended**: 40GB+ RAM, CUDA-compatible GPU 
- **Storage**: 50GB+ free space for model downloads

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. Please check individual model licenses for specific usage terms.

## Acknowledgments

- Hugging Face Transformers library
- Gradio for the user interface
- All model creators and maintainers
- Docling team for document processing capabilities

> [!important]
The community GPU grant was given by [Hugging Face](https://huggingface.co/prithivMLmods) — special thanks to them. 🤗🚀

## Support

For issues and questions, please open an issue on the GitHub repository.
