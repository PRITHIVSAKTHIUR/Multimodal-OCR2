import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

from docling_core.types.doc import DoclingDocument, DocTagsDocument

import re
import ast
import html

# Constants for text generation
MAX_MAX_NEW_TOKENS = 5120
DEFAULT_MAX_NEW_TOKENS = 3072
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Nanonets-OCR-s
MODEL_ID_M = "nanonets/Nanonets-OCR-s"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load MonkeyOCR
MODEL_ID_G = "echo840/MonkeyOCR"
SUBFOLDER = "Recognition"
processor_g = AutoProcessor.from_pretrained(
    MODEL_ID_G,
    trust_remote_code=True,
    subfolder=SUBFOLDER
)
model_g = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_G,
    trust_remote_code=True,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16
).to(device).eval()

# Load Typhoon-OCR-7B
MODEL_ID_L = "scb10x/typhoon-ocr-7b"
processor_l = AutoProcessor.from_pretrained(MODEL_ID_L, trust_remote_code=True)
model_l = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_L,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load SmolDocling-256M-preview
MODEL_ID_X = "ds4sd/SmolDocling-256M-preview"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Thyme-RL
MODEL_ID_N = "Kwai-Keye/Thyme-RL"
processor_n = AutoProcessor.from_pretrained(MODEL_ID_N, trust_remote_code=True)
model_n = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_N,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Preprocessing functions for SmolDocling-256M
def add_random_padding(image, min_percent=0.1, max_percent=0.10):
    """Add random padding to an image based on its size."""
    image = image.convert("RGB")
    width, height = image.size
    pad_w_percent = random.uniform(min_percent, max_percent)
    pad_h_percent = random.uniform(min_percent, max_percent)
    pad_w = int(width * pad_w_percent)
    pad_h = int(height * pad_h_percent)
    corner_pixel = image.getpixel((0, 0))  # Top-left corner
    padded_image = ImageOps.expand(image, border=(pad_w, pad_h, pad_w, pad_h), fill=corner_pixel)
    return padded_image

def normalize_values(text, target_max=500):
    """Normalize numerical values in text to a target maximum."""
    def normalize_list(values):
        max_value = max(values) if values else 1
        return [round((v / max_value) * target_max) for v in values]

    def process_match(match):
        num_list = ast.literal_eval(match.group(0))
        normalized = normalize_list(num_list)
        return "".join([f"<loc_{num}>" for num in normalized])

    pattern = r"\[([\d\.\s,]+)\]"
    normalized_text = re.sub(pattern, process_match, text)
    return normalized_text

def downsample_video(video_path):
    """Downsample a video to evenly spaced frames, returning PIL images with timestamps."""
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """Generate responses for image input using the selected model."""
    if model_name == "Nanonets-OCR-s":
        processor = processor_m
        model = model_m
    elif model_name == "MonkeyOCR-Recognition":
        processor = processor_g
        model = model_g
    elif model_name == "SmolDocling-256M-preview":
        processor = processor_x
        model = model_x
    elif model_name == "Typhoon-OCR-7B":
        processor = processor_l
        model = model_l
    elif model_name == "Thyme-RL":
        processor = processor_n
        model = model_n
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    images = [image]

    if model_name == "SmolDocling-256M-preview":
        if "OTSL" in text or "code" in text:
            images = [add_random_padding(img) for img in images]
        if "OCR at text at" in text or "Identify element" in text or "formula" in text:
            text = normalize_values(text, target_max=500)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [
                {"type": "text", "text": text}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text.replace("<|im_end|>", "")
        yield buffer, buffer

    if model_name == "SmolDocling-256M-preview":
        cleaned_output = buffer.replace("<end_of_utterance>", "").strip()
        if any(tag in cleaned_output for tag in ["<doctag>", "<otsl>", "<code>", "<chart>", "<formula>"]):
            if "<chart>" in cleaned_output:
                cleaned_output = cleaned_output.replace("<chart>", "<otsl>").replace("</chart>", "</otsl>")
                cleaned_output = re.sub(r'(<loc_500>)(?!.*<loc_500>)<[^>]+>', r'\1', cleaned_output)
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([cleaned_output], images)
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            markdown_output = doc.export_to_markdown()
            yield buffer, markdown_output
        else:
            yield buffer, cleaned_output

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """Generate responses for video input using the selected model."""
    if model_name == "Nanonets-OCR-s":
        processor = processor_m
        model = model_m
    elif model_name == "MonkeyOCR-Recognition":
        processor = processor_g
        model = model_g
    elif model_name == "SmolDocling-256M-preview":
        processor = processor_x
        model = model_x
    elif model_name == "Typhoon-OCR-7B":
        processor = processor_l
        model = model_l
    elif model_name == "Thyme-RL":
        processor = processor_n
        model = model_n
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    images = [frame for frame, _ in frames]

    if model_name == "SmolDocling-256M-preview":
        if "OTSL" in text or "code" in text:
            images = [add_random_padding(img) for img in images]
        if "OCR at text at" in text or "Identify element" in text or "formula" in text:
            text = normalize_values(text, target_max=500)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [
                {"type": "text", "text": text}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text.replace("<|im_end|>", "")
        yield buffer, buffer

    if model_name == "SmolDocling-256M-preview":
        cleaned_output = buffer.replace("<end_of_utterance>", "").strip()
        if any(tag in cleaned_output for tag in ["<doctag>", "<otsl>", "<code>", "<chart>", "<formula>"]):
            if "<chart>" in cleaned_output:
                cleaned_output = cleaned_output.replace("<chart>", "<otsl>").replace("</chart>", "</otsl>")
                cleaned_output = re.sub(r'(<loc_500>)(?!.*<loc_500>)<[^>]+>', r'\1', cleaned_output)
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([cleaned_output], images)
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            markdown_output = doc.export_to_markdown()
            yield buffer, markdown_output
        else:
            yield buffer, cleaned_output

# Define examples for image and video inference
image_examples = [
    ["Reconstruct the doc [table] as it is.", "images/0.png"],
    ["Describe the image!", "images/8.png"],
    ["OCR the image", "images/2.jpg"],
    ["Convert this page to docling", "images/1.png"],
    ["Convert this page to docling", "images/3.png"],
    ["Convert chart to OTSL.", "images/4.png"],
    ["Convert code to text", "images/5.jpg"],
    ["Convert this table to OTSL.", "images/6.jpg"],
    ["Convert formula to late.", "images/7.jpg"],
]

video_examples = [
    ["Explain the video in detail.", "videos/1.mp4"],
    ["Explain the video in detail.", "videos/2.mp4"]
]

#css
css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Multimodal OCR2](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image", height=290)
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video", height=290)
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
                
        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Output")
                raw_output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=5)
                
                with gr.Accordion("(Result.md)", open=False):
                    formatted_output = gr.Markdown(label="(Result.md)")
            
            model_choice = gr.Radio(
                choices=["Nanonets-OCR-s", "MonkeyOCR-Recognition", "Thyme-RL", "Typhoon-OCR-7B", "SmolDocling-256M-preview"],
                label="Select Model",
                value="Nanonets-OCR-s"
            )
            
            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR2/discussions)")
            gr.Markdown("> [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s): nanonets-ocr-s is a powerful, state-of-the-art image-to-markdown ocr model that goes far beyond traditional text extraction. it transforms documents into structured markdown with intelligent content recognition and semantic tagging.")
            gr.Markdown("> [SmolDocling-256M](https://huggingface.co/ds4sd/SmolDocling-256M-preview): SmolDocling is a multimodal Image-Text-to-Text model designed for efficient document conversion. It retains Docling's most popular features while ensuring full compatibility with Docling through seamless support for DoclingDocuments.")
            gr.Markdown("> [MonkeyOCR-Recognition](https://huggingface.co/echo840/MonkeyOCR): MonkeyOCR adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.")
            gr.Markdown("> [Typhoon-OCR-7B](https://huggingface.co/scb10x/typhoon-ocr-7b): A bilingual document parsing model built specifically for real-world documents in Thai and English inspired by models like olmOCR based on Qwen2.5-VL-Instruction. Extracts and interprets embedded text (e.g., chart labels, captions) in Thai or English.")
            gr.Markdown("> [Thyme-RL](https://huggingface.co/Kwai-Keye/Thyme-RL): Thyme: Think Beyond Images. Thyme transcends traditional ``thinking with images'' paradigms by autonomously generating and executing diverse image processing and computational operations through executable code, significantly enhancing performance on high-resolution perception and complex reasoning tasks.")
            gr.Markdown(">⚠️note: all the models in space are not guaranteed to perform well in video inference use cases.")  
            
    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[raw_output, formatted_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[raw_output, 
                 formatted_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)
