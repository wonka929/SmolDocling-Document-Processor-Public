# SmolDocling Document Processor

SmolDocling Document Processor is a lightweight application that processes document images or PDFs and converts them into structured formats such as Markdown, HTML, or JSON. It uses the Hugging Face Transformers version of SmolDocling and runs on CUDA when an Nvidia GPU is available.

## Features

- Upload PDF or image files.
- Provide a URL to a remote PDF or image.
- Export results as Markdown, HTML, or JSON.
- Preserve document structure through DocTags and Docling export utilities.

## Installation

1. Clone the repository.

```bash
git clone https://github.com/bibekplus/SmolDocling-Document-Processor.git
cd SmolDocling-Document-Processor
```

2. Install dependencies.

```bash
uv sync
```

Or, if you prefer pip:

```bash
pip install -r requirements.txt
```

3. Verify CUDA support if you want GPU acceleration.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Usage

```bash
python main.py
```

Then open the Gradio interface in your browser, upload a document or provide a URL, choose the export format, and start processing.

## Requirements

- Python 3.11 or higher.
- Dependencies listed in [pyproject.toml](pyproject.toml) or [requirements.txt](requirements.txt).
- For GPU acceleration, a CUDA-enabled PyTorch installation and an Nvidia GPU.

## Notes

- The application automatically selects CUDA when available.
- If CUDA is unavailable, inference falls back to CPU.
- SmolDocling is a compact Visual Language Model for document understanding.

## License

This project is licensed under the MIT License. See the LICENSE file for details.# SmolDocling Document Processor

SmolDocling Document Processor is a lightweight application that processes document images or PDFs and converts them into structured formats such as Markdown, HTML, or JSON. It leverages the **SmolDocling** Visual Language Model (VLM) for document understanding, making it ideal for extracting semantic meaning from diverse document types.

---

## Features

- **Input Options**:
  - Upload PDF or image files.
  - Provide a URL to a remote PDF or image.

- **Output Formats**:
  - **Markdown**: For easy viewing and copy-pasting.
  - **HTML**: Preserves rich layout for web rendering.
  - **JSON**: Ideal for developers and downstream automation.

- **Capabilities**:
  - Understands full pages of diverse document types (e.g., academic papers, business forms, patents).
  - Extracts:
    - Paragraphs, headers, and footers.
    - Tables and their structure (including merged cells and headers).
    - Code blocks with indentation.
    - Equations (LaTeX format).
    - Charts and captions.
    - Lists and nested lists.
  - Maintains spatial layout and reading order.
  - Outputs results in structured **DocTags**, convertible into Markdown, HTML, and JSON.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bibekplus/SmolDocling-Document-Processor.git
   cd SmolDocling-Document-Processor

2. Install the required dependencies:

    ```bash
    uv sync
    ```
    or

    ```bash
    pip install -r requirements.txt

3. Ensure you have the necessary backend setup for MLX (optimized for Apple Silicon but works on other platforms).

<hr>

## Usage
1. Run the application:

   ```bash
   python main.py


2. Open the Gradio interface in your browser.

3. Upload a document or provide a URL, select the desired output format, and click Process Document.

4. View the structured output, preview the document, and download the results.

## Requirements
 - Python 3.11 or higher.
 - Dependencies listed in requirements.txt (e.g., gradio, torch, pdf2image, mlx-vlm).

## Notes
- This app is optimized for Apple Silicon (Metal backend with MLX) but works on other machines with appropriate setup.
- SmolDocling is a compact (256MB) Visual Language Model designed for document understanding.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
