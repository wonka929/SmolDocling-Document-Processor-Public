import importlib.util
import json
import os
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import requests
import torch
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def load_input_resource(input_path):
    """Load an image or PDF from a local path or URL and return page images."""
    images = []

    if urlparse(input_path).scheme:
        response = requests.get(input_path, stream=True, timeout=10)
        response.raise_for_status()
        content = BytesIO(response.content)

        content.seek(0)
        if content.read(4) == b"%PDF":
            content.seek(0)
            images.extend(convert_from_bytes(content.read()))
        else:
            content.seek(0)
            images.append(Image.open(content).convert("RGB"))
    else:
        file_path = Path(input_path)
        if file_path.suffix.lower() == ".pdf":
            images.extend(convert_from_path(str(file_path)))
        else:
            images.append(Image.open(file_path).convert("RGB"))

    return images


def resolve_torch_device():
    """Select CUDA when available, otherwise fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(device):
    """Pick the most suitable dtype for the active device."""
    if device != "cuda":
        return torch.float32
    return torch.float16


def resolve_attention_implementation(device):
    """Forza 'eager' per compatibilità con Idefics3VisionTransformer."""
    return "eager"


def clean_doctags_output(output_text):
    """Remove trailing control tokens and keep the final DocTags payload only."""
    cleaned_output = output_text.replace("<end_of_utterance>", "").strip()
    if "</doctag>" in cleaned_output:
        cleaned_output = cleaned_output.split("</doctag>", 1)[0] + "</doctag>"
    return cleaned_output


@lru_cache(maxsize=1)
def load_model():
    """Load the Hugging Face SmolDocling model with a CUDA-first configuration."""
    model_path = "docling-project/SmolDocling-256M-preview"
    device = resolve_torch_device()
    torch_dtype = resolve_torch_dtype(device)
    attn_implementation = resolve_attention_implementation(device)

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        _attn_implementation=attn_implementation,
    ).to(device)
    model.eval()

    print(f"Running SmolDocling on {device} with dtype={torch_dtype}")
    return model, processor, device


def generate_doctags(model, processor, device, prompt, image):
    """Run one image through the model and return its DocTags output."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=formatted_prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,
        )

    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    output = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    return clean_doctags_output(output)


def process_document(file_obj, url_input, export_format):
    """Process a document with SmolDocling and return the results."""
    try:
        model, processor, device = load_model()

        if file_obj is not None:
            temp_dir = tempfile.mkdtemp()
            file_name = Path(getattr(file_obj, "name", "uploaded_file")).name
            temp_path = os.path.join(temp_dir, file_name)

            if hasattr(file_obj, "read"):
                with open(temp_path, "wb") as handle:
                    handle.write(file_obj.read())
            elif isinstance(file_obj, str):
                temp_path = file_obj
            else:
                temp_path = file_obj.name

            input_path = temp_path
        elif url_input.strip():
            input_path = url_input.strip()
        else:
            return "Please provide either a file upload or a URL", None, None

        images = load_input_resource(input_path)
        if not images:
            return "No images could be extracted from the provided file or URL", None, None

        prompt = "Convert this page to docling."
        all_outputs = []
        all_images = []
        processing_log = ""

        for index, image in enumerate(images, start=1):
            processing_log += f"Processing page {index}/{len(images)}...\n\n"
            processing_log += "DocTags:\n\n"

            all_images.append(image)
            output = generate_doctags(model, processor, device, prompt, image)
            all_outputs.append(output)
            processing_log += output + "\n\n"

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_outputs, all_images)
        doc = DoclingDocument(name="ProcessedDocument")
        doc.load_from_doctags(doctags_doc)

        if export_format == "Markdown":
            result = doc.export_to_markdown()
        elif export_format == "HTML":
            html_output = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            html_path = Path(html_output.name)
            doc.save_as_html(html_path, image_mode=ImageRefMode.EMBEDDED)
            with open(html_path, "r", encoding="utf-8") as handle:
                result = handle.read()
        elif export_format == "JSON":
            result = json.dumps(doc.export_to_dict(), indent=4)
        else:
            result = "Invalid export format selected"

        return result, images[0] if images else None, processing_log
    except Exception as exc:
        import traceback

        error_details = traceback.format_exc()
        return f"Error processing document: {exc}\n\nDetails:\n{error_details}", None, error_details


def render_output(result, export_format):
    """Render the processed result based on export format."""
    if export_format == "Markdown":
        return gr.update(value=result, visible=True), gr.update(visible=False), gr.update(visible=False)
    if export_format == "HTML":
        return gr.update(visible=False), gr.update(value=result, visible=True), gr.update(visible=False)
    if export_format == "JSON":
        try:
            json_obj = json.loads(result)
        except Exception as exc:
            json_obj = {"error": "Invalid JSON", "detail": str(exc)}
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=json_obj, visible=True)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def prepare_download(result, export_format):
    """Prepare a downloadable file for the processed output."""
    if export_format == "Markdown":
        extension = ".md"
    elif export_format == "HTML":
        extension = ".html"
    elif export_format == "JSON":
        extension = ".json"
    else:
        extension = ".txt"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(result.encode("utf-8"))
    temp_file.close()
    return gr.update(value=temp_file.name), gr.update(value=temp_file.name)


with gr.Blocks(title="SmolDocling Document Processing") as app:
    gr.HTML(
        """
        <style>
        #raw_output_box, #formatted_output_box {
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """
    )

    gr.Markdown(
        """
    # SmolDocling Document Processor

    Upload a document image or PDF, or provide a URL, to convert it into a structured format using SmolDocling.

    SmolDocling is a compact (256MB) Visual Language Model designed for document understanding. It can analyze document layouts,
    identify structural elements, and generate structured representations that preserve the document's semantic meaning.
    """
    )

    lang = None
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF or Image")
            url_input = gr.Textbox(label="Or enter a URL to a PDF or Image")
            export_format = gr.Radio(
                choices=["Markdown", "HTML", "JSON"],
                label="Export Format",
                value="Markdown",
            )
            submit_button = gr.Button("Process Document", variant="primary")
        if export_format == "Markdown":
            lang = "markdown"
        elif export_format == "HTML":
            lang = "html"
        elif export_format == "JSON":
            lang = "json"
        with gr.Column(scale=2):
            with gr.Tab("Raw Output"):
                with gr.Column(elem_id="raw_output_box"):
                    output_text = gr.Code(label="Structured Output", language=lang, lines=20, max_lines=20)
                    download_raw = gr.DownloadButton("Download Raw Output")
            with gr.Tab("Document Preview"):
                preview_image = gr.Image(label="Document Preview", type="pil")
            with gr.Tab("Log"):
                log_output = gr.Code(label="Processing Log", language="html", lines=20, max_lines=20)
            with gr.Tab("Formatted Output"):
                with gr.Column(elem_id="formatted_output_box"):
                    rendered_markdown = gr.Markdown(visible=False, label="Markdown Render")
                    rendered_html = gr.HTML(visible=False, label="HTML Render")
                    rendered_json = gr.JSON(visible=False, label="JSON Render")
                    download_formatted = gr.DownloadButton("Download Formatted Output")

    gr.Markdown(
        """
    ### What SmolDocling can do

    - Understand full pages of diverse document types: academic papers, business forms, patents, and more.
    - Extract paragraphs, headers, footers, tables, code blocks, equations, charts, captions, and lists.
    - Maintain spatial layout and reading order.
    - Output structured DocTags that can be exported as Markdown, HTML, or JSON.

    ### Input options

    - Upload PDF or image files.
    - Enter a URL to a remote PDF or image.

    ### Output formats

    - Markdown for easy viewing and copy-pasting.
    - HTML for rich rendering.
    - JSON for downstream automation.

    Note: this app uses Transformers with PyTorch and prefers CUDA on Nvidia GPUs. If CUDA is unavailable, it falls back to CPU.
    """
    )

    submit_button.click(
        process_document,
        inputs=[file_input, url_input, export_format],
        outputs=[output_text, preview_image, log_output],
    ).then(
        render_output,
        inputs=[output_text, export_format],
        outputs=[rendered_markdown, rendered_html, rendered_json],
    ).then(
        prepare_download,
        inputs=[output_text, export_format],
        outputs=[download_raw, download_formatted],
    )


if __name__ == "__main__":
    app.launch()
