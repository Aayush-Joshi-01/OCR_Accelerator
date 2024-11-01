# OCR Accelerator - `ocr_models.py`

## Overview

The `ocr_models.py` module manages the available OCR models and their configurations. It supports multiple models such as Microsoft TrOCR, Tesseract, LlamaParse, EasyOCR, Llama Vision, and Pixtral. This module is responsible for initializing and processing images using the selected OCR model.

## Features

- **Model Configuration**: Defines configurations for various OCR models.
- **Model Initialization**: Initializes the selected OCR model with the required components.
- **Image Processing**: Processes images using the selected OCR model and document type.
- **Preprocessing**: Applies preprocessing steps to the images for enhanced OCR accuracy.
- **Formatting**: Formats the extracted text according to document type specifications.

## Key Components

### `ModelConfig` Dataclass

The `ModelConfig` dataclass stores configuration details for each OCR model.

#### Attributes

- **`name`**: The name of the OCR model.
- **`description`**: A brief description of the OCR model.
- **`supports_document_types`**: List of document types supported by the model.
- **`api_config`**: Optional dictionary for API configuration.

```python
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    description: str
    supports_document_types: list[str]
    api_config: Optional[Dict[str, Any]] = None
```

### `get_available_models` Function

The `get_available_models` function returns a dictionary of available OCR models with their configurations.

```python
def get_available_models() -> Dict[str, ModelConfig]:
    """
    Returns a dictionary of available OCR models with their configurations.
    """
    return {
        'microsoft_trocr': ModelConfig(
            name='Microsoft TrOCR',
            description='Transformer-based OCR model optimized for handwritten text',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'model_name': 'microsoft/trocr-base-handwritten'
            }
        ),
        'tesseract': ModelConfig(
            name='Tesseract OCR',
            description='Traditional OCR using Tesseract engine',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'lang': 'eng',
                'config': '--psm 6'
            }
        ),
        'llama_parse_gemini_pro': ModelConfig(
            name='LlamaParse with Gemini Pro',
            description='Uses LlamaParse with Gemini Pro for advanced document understanding',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'vendor_multimodal_model_name': 'gemini-1.5-pro',
                'result_type': 'markdown'
            }
        ),
        'llama_parse_gemini_flash': ModelConfig(
            name='LlamaParse with Gemini Flash',
            description='Uses LlamaParse with Gemini Flash for fast document understanding',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'vendor_multimodal_model_name': 'gemini-1.5-flash',
                'result_type': 'markdown'
            }
        ),
        'llama_parse_gpt4_mini': ModelConfig(
            name='LlamaParse with GPT-4-mini',
            description='Uses LlamaParse with GPT-4-mini for document understanding',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'vendor_multimodal_model_name': 'openai-gpt-4o-mini',
                'result_type': 'markdown'
            }
        ),
        'easyocr': ModelConfig(
            name='EasyOCR',
            description='Simple and accurate OCR tool with multiple language support',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'langs': ['en'],
                'gpu': False
            }
        ),
        'llama_vision': ModelConfig(
            name='Llama Vision 3.2 90B',
            description='Advanced vision model for document understanding and text extraction',
            supports_document_types=['log_notes', 'legal_documents', 'medical_records', 'financial_records', 'general_config'],
            api_config={
                'model_name': 'llama-3.2-90b-vision-preview'
            }
        ),
        'pixtral': ModelConfig(
            name='Pixtral 12B',
            description='Advanced vision-language model for document understanding and structure preservation',
            supports_document_types=['log_notes', 'legal_documents', 'financial_recor', 'medical_records', 'general_config'],
            api_config={
                'model_name': 'pixtral-12b-2409'
            }
        )
    }
```

### `OCRModelManager` Class

The `OCRModelManager` class manages the initialization and processing of OCR models.

#### Methods

- **`__init__`**: Initializes the OCR Model Manager.
- **`process_image`**: Processes the image with the selected OCR model and document configuration.
- **`_initialize_model_components`**: Initializes model-specific components.
- **`cleanup_text`**: Cleans up the extracted text.
- **`_load_pdf_data`**: Loads PDF data using LlamaParse.
- **`_process_with_microsoft_trocr`**: Processes the image using Microsoft TrOCR.
- **`_process_with_easyocr`**: Processes the image using EasyOCR.
- **`_process_with_llama_vision`**: Processes the image using Llama Vision.
- **`_process_with_tesseract`**: Processes the image using Tesseract OCR.
- **`_process_with_llama_parse`**: Processes the image using LlamaParse.
- **`_process_with_pixtral`**: Processes the image using Pixtral.
- **`_format_extracted_text`**: Formats the extracted text according to document type specifications.

```python
import logging
from PIL import Image
import cv2
import numpy as np
import pytesseract
import asyncio
import nest_asyncio
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from groq import Groq
from mistralai import Mistral
from easyocr import Reader
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from document_configs import get_document_configs

logger = logging.getLogger(__name__)

class OCRModelManager:
    def __init__(
        self,
        model_key: str,
        document_type: str,
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY"),
        mistral_api_key: Optional[str] = os.getenv("MISTRAL_KEY")
    ):
        """
        Initialize the OCR Model Manager.

        Args:
            model_key: Key of the model to use
            document_type: Type of document being processed
            groq_api_key: API key for Groq (required for Llama Vision)
            mistral_api_key: API key for Mistral (required for Pixtral)
        """
        nest_asyncio.apply()
        self.model_key = model_key
        self.document_type = document_type
        self.groq_api_key = groq_api_key
        self.mistral_api_key = mistral_api_key
        self.models = get_available_models()
        self.document_configs = get_document_configs()

        # Validate inputs
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}")
        if document_type not in self.document_configs:
            raise ValueError(f"Unknown document type: {document_type}")

        self.model_config = self.models[model_key]
        self.doc_config = self.document_configs[document_type]

        if document_type not in self.model_config.supports_document_types:
            raise ValueError(f"Model {model_key} does not support document type {document_type}")

        # Initialize model-specific components
        self._initialize_model_components()

    def process_image(self, image: Image.Image) -> str:
        """
        Process the image with the selected OCR model and document configuration.
        """
        logger.info(f"Processing {self.document_type} with {self.model_key}")

        try:
            if self.model_key.startswith('llama_parse'):
                return self._process_with_llama_parse(image)
            elif self.model_key == 'tesseract':
                return self._process_with_tesseract(image)
            elif self.model_key == 'microsoft_trocr':
                return self._process_with_microsoft_trocr(image)
            elif self.model_key == 'easyocr':
                return self._process_with_easyocr(image)
            elif self.model_key == 'llama_vision':
                return self._process_with_llama_vision(image)
            elif self.model_key == 'pixtral':
                return self._process_with_pixtral(image)
            else:
                raise ValueError(f"Unsupported model: {self.model_key}")
        except Exception as e:
            logger.error(f"Error processing image with {self.model_key}: {str(e)}")
            raise

    def _initialize_model_components(self):
        """Initialize model components that can be reused across calls"""
        self.components = {}

        try:
            if self.model_key == 'microsoft_trocr':
                self.components['processor'] = TrOCRProcessor.from_pretrained(
                    self.model_config.api_config['model_name']
                )
                self.components['model'] = VisionEncoderDecoderModel.from_pretrained(
                    self.model_config.api_config['model_name']
                )

            elif self.model_key == 'easyocr':
                self.components['reader'] = Reader(
                    self.model_config.api_config['langs'],
                    gpu=self.model_config.api_config['gpu']
                )

            elif self.model_key == 'llama_vision' and self.groq_api_key:
                self.components['client'] = Groq(api_key=self.groq_api_key)

            elif self.model_key == 'pixtral' and self.mistral_api_key:
                self.components['client'] = Mistral(api_key=self.mistral_api_key)

        except Exception as e:
            logger.error(f"Error initializing model components: {str(e)}")
            raise

    @staticmethod
    def cleanup_text(text: str) -> str:
        """Clean up text by removing non-ASCII characters"""
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    async def _load_pdf_data(self, file_extractor: Dict, input_path: str) -> List:
        """Load PDF data using LlamaParse"""
        reader = SimpleDirectoryReader(
            input_files=[input_path],
            file_extractor=file_extractor
        )
        documents = await reader.aload_data()
        return documents

    def _process_with_microsoft_trocr(self, image: Image.Image) -> str:
        """Process image using Microsoft's TrOCR model"""
        try:
            # Convert image to RGB if needed
            image = image.convert("RGB")

            # Get pixel values
            pixel_values = self.components['processor'](
                image,
                return_tensors="pt"
            ).pixel_values

            # Generate text
            generated_ids = self.components['model'].generate(pixel_values)
            generated_text = self.components['processor'].batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return self._format_extracted_text(generated_text)
        except Exception as e:
            raise Exception(f"Error processing image with TrOCR: {str(e)}")

    def _process_with_easyocr(self, image: Image.Image) -> str:
        """Process image using EasyOCR"""
        try:
            # Convert PIL Image to cv2 format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Perform OCR
            results = self.components['reader'].readtext(image_cv)

            # Extract and clean text
            text_string = " ".join([self.cleanup_text(text[1]) for text in results])
            return self._format_extracted_text(text_string)

        except Exception as e:
            raise Exception(f"Error processing with EasyOCR: {str(e)}")

    def _process_with_llama_vision(self, image: Image.Image) -> str:
        """Process image using Llama Vision 3.2 90B"""
        if not self.groq_api_key:
            raise ValueError("Groq API key is required for Llama Vision processing")

        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Create chat completion request
            chat_completion = self.components['client'].chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.doc_config.ocr_instruction
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                model=self.model_config.api_config['model_name']
            )

            return self._format_extracted_text(
                chat_completion.choices[0].message.content
            )

        except Exception as e:
            raise Exception(f"Error processing with Llama Vision: {str(e)}")

    def _process_with_tesseract(self, image: Image.Image) -> str:
        """Process image using Tesseract OCR"""
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.model_config.api_config['lang'],
                config=self.model_config.api_config['config']
            )

            return self._format_extracted_text(text)
        except Exception as e:
            raise Exception(f"Error processing image with Tesseract: {str(e)}")

    def _process_with_llama_parse(self, image: Image.Image) -> str:
        """Process image using LlamaParse with configured model"""
        try:
            # Save image as temporary PDF
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = os.path.join(tmp_dir, "temp.pdf")
                image.save(pdf_path, "PDF", resolution=300, save_all=True)

                # Initialize LlamaParse with model configuration
                parser = LlamaParse(
                    parsing_instruction=self.doc_config.ocr_instruction,
                    result_type=self.model_config.api_config['result_type'],
                    use_vendor_multimodal_model=True,
                    vendor_multimodal_model_name=self.model_config.api_config['vendor_multimodal_model_name']
                )

                # Process document
                file_extractor = {".pdf": parser}
                documents = asyncio.run(self._load_pdf_data(file_extractor, pdf_path))

                return self._format_extracted_text(documents[0].text)

        except Exception as e:
            raise Exception(f"Error processing with LlamaParse: {str(e)}")

    def _process_with_pixtral(self, image: Image.Image) -> str:
        """Process image using Pixtral 12B model"""
        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required for Pixtral processing")

        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare messages for the chat
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.doc_config.ocr_instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Get the chat response
            chat_response = self.components['client'].chat.complete(
                model=self.model_config.api_config['model_name'],
                messages=messages
            )

            return self._format_extracted_text(
                chat_response.choices[0].message.content
            )

        except Exception as e:
            raise Exception(f"Error processing with Pixtral: {str(e)}")

    def _format_extracted_text(self, text: str) -> str:
        """
        Format the extracted text according to document type specifications.
        """
        # Apply document-specific formatting rules
        if hasattr(self.doc_config, 'text_formatter'):
            return self.doc_config.text_formatter(text)

        # Default formatting if no specific formatter is defined
        formatted_text = text.strip()
        formatted_text = "\n".join(line.strip() for line in formatted_text.split("\n") if line.strip())
        return f"# {self.doc_config.name}\n\n{formatted_text}"
```

### `process_image_with_model` Function

The `process_image_with_model` function processes the image with the selected OCR model and document type.

```python
def process_image_with_model(
    image: Image.Image,
    model_key: str,
    document_type: str
) -> str:
    """
    Process the image with the selected OCR model and document type.

    Args:
        image: PIL Image object
        model_key: Key of the selected model
        document_type: Type of document being processed

    Returns:
        str: Processed text in markdown format
    """
    processor = OCRModelManager(model_key, document_type)
    return processor.process_image(image)
```

## Usage

The `ocr_models.py` module is used to manage and process images using various OCR models. This module is integrated into the main application to handle the OCR processing based on the selected model and document type.

### Example Usage

```python
from ocr_models import process_image_with_model
from PIL import Image

# Load an image
image = Image.open("path/to/your/image.jpg")

# Process the image with a specific OCR model and document type
result = process_image_with_model(image, 'tesseract', 'log_notes')

# Print the processed result
print(result)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
