# OCR Accelerator

## Overview

The OCR Accelerator is a Streamlit-based application designed to process and analyze various types of documents using Optical Character Recognition (OCR) and Large Language Models (LLMs). This project supports multiple document types and OCR models, allowing for flexible and powerful document processing capabilities.

## Features

- **Document Type Selection**: Choose from different document types such as log notes, medical records, legal documents, and financial records.
- **Model Selection**: Select from a variety of OCR models including Microsoft TrOCR, Tesseract, LlamaParse, EasyOCR, Llama Vision, and Pixtral.
- **Preprocessing Controls**: Adjust image preprocessing parameters to enhance OCR accuracy.
- **AI Conclusion**: Generate AI-based conclusions for processed documents.
- **Download Results**: Download processed results in Markdown or Text format.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Streamlit
- Pillow
- OpenCV
- PyTesseract
- Transformers
- Groq
- Mistralai
- EasyOCR
- LlamaParse
- Google Generative AI

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aayush-Joshi-01/OCR_Accelerator.git
   cd OCR_Accelerator
   ```

2. **Create and activate a virtual environment**:
    On Windows:

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

    On Mac OS and Linux:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
    GEMINI_API_KEY=
    LLAMA_CLOUD_API_KEY=
    GROQ_API_KEY=
    MISTRAL_KEY=
   ```
5. **Verify installation and configuration**:
    ```bash
    python -c "import streamlit; print(streamlit.__version__)"
    ```
    This should print the installed Streamlit version without any errors.

Now your virtual environment is set up with all the required dependencies.

## Usage

### Running the Application

To run the OCR Accelerator application, execute the following command:
```bash
streamlit run app.py
```

### Using the Application

1. **Document Type Selection**:
   - Select the type of document you want to process from the sidebar.

2. **Model Selection**:
   - Choose an OCR model from the available options.

3. **Preprocessing Controls**:
   - Enable and adjust preprocessing parameters to improve OCR results.

4. **Upload Document**:
   - Upload the document image you want to process.

5. **Process Document**:
   - Click the "Process Document" button to start the OCR process.

6. **View Results**:
   - The processed text and AI conclusions (if enabled) will be displayed.

7. **Download Results**:
   - Choose the download format (Markdown or Text) and click the "Download Results" button.

## Modules

### `app.py`

The main application file that integrates all components and provides the user interface using Streamlit.

#### Key Components

- **OCRApp Class**: Manages the application state and integrates various functionalities.
- **setup_page_config**: Configures the Streamlit page settings.
- **initialize_session_state**: Initializes the session state for various parameters.
- **render_document_type_selection**: Renders the document type selection UI.
- **render_model_selection**: Renders the model selection UI.
- **process_image**: Processes the uploaded image using the selected OCR model.
- **show_preprocessing_controls**: Displays preprocessing controls for image enhancement.
- **preprocess_image**: Applies preprocessing steps to the image.
- **render_processing_controls**: Renders the processing controls UI.
- **check_regeneration**: Checks if the selected model and document type require regeneration.
- **process_image**: Processes the image and optionally generates AI conclusions.
- **download_results**: Handles downloading the processed results.
- **render_processing_result**: Renders the processed results and AI conclusions.
- **run**: Main function to run the application.

### `document_configs.py`

Defines configurations for different document types, including OCR instructions, LLM instructions, preprocessing defaults, and recommended models.

#### Key Components

- **DocumentConfig Dataclass**: Stores configuration details for each document type.
- **get_document_configs**: Returns a dictionary of document configurations.

### `llm.py`

Handles interactions with the Large Language Model (LLM) for generating AI conclusions. Uses the Google Generative AI library.

#### Key Components

- **restructure_response**: Generates a structured response using the LLM.
- **regen_document_types**: List of document types that require regeneration.
- **regen_model_types**: List of model types that require regeneration.

### `ocr_models.py`

Manages the available OCR models and their configurations. Supports multiple models such as Microsoft TrOCR, Tesseract, LlamaParse, EasyOCR, Llama Vision, and Pixtral.

#### Key Components

- **ModelConfig Dataclass**: Stores configuration details for each OCR model.
- **get_available_models**: Returns a dictionary of available OCR models.
- **OCRModelManager Class**: Manages the initialization and processing of OCR models.
- **process_image**: Processes the image using the selected OCR model.
- **_initialize_model_components**: Initializes model-specific components.
- **cleanup_text**: Cleans up the extracted text.
- **_load_pdf_data**: Loads PDF data using LlamaParse.
- **_process_with_microsoft_trocr**: Processes the image using Microsoft TrOCR.
- **_process_with_easyocr**: Processes the image using EasyOCR.
- **_process_with_llama_vision**: Processes the image using Llama Vision.
- **_process_with_tesseract**: Processes the image using Tesseract OCR.
- **_process_with_llama_parse**: Processes the image using LlamaParse.
- **_process_with_pixtral**: Processes the image using Pixtral.
- **_format_extracted_text**: Formats the extracted text according to document type specifications.
- **process_image_with_model**: Processes the image with the selected OCR model and document type.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
