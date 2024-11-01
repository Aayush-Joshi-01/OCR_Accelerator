# OCR Accelerator - `document_configs.py`

## Overview

The `document_configs.py` module defines configurations for different document types, including OCR instructions, LLM instructions, preprocessing defaults, and recommended models. This module is essential for tailoring the OCR and LLM processing to specific document types, ensuring accurate and relevant results.

## Features

- **Document Type Configurations**: Defines configurations for various document types such as log notes, medical records, legal documents, and financial records.
- **OCR Instructions**: Provides specific instructions for OCR processing tailored to each document type.
- **LLM Instructions**: Provides specific instructions for Large Language Model (LLM) processing tailored to each document type.
- **Preprocessing Defaults**: Sets default preprocessing parameters for each document type to enhance OCR accuracy.
- **Recommended Models**: Lists recommended OCR models for each document type.

## Key Components

### `DocumentConfig` Dataclass

The `DocumentConfig` dataclass stores configuration details for each document type.

#### Attributes

- **`name`**: The name of the document type.
- **`description`**: A brief description of the document type.
- **`ocr_instruction`**: Specific instructions for OCR processing.
- **`llm_instruction`**: Specific instructions for LLM processing.
- **`preprocessing_defaults`**: Default preprocessing parameters.
- **`recommended_models`**: List of recommended OCR models for the document type.

```python
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentConfig:
    name: str
    description: str
    ocr_instruction: str
    llm_instruction: str
    preprocessing_defaults: Dict[str, Any]
    recommended_models: list[str]
```

### `get_document_configs` Function

The `get_document_configs` function returns a dictionary of document configurations.

```python
def get_document_configs() -> Dict[str, DocumentConfig]:
    """
    Returns configurations for different document types.
    """
    return {
        'log_notes': DocumentConfig(
            name="Log Notes",
            description="Handwritten or typed log notes and records",
            ocr_instruction="""This document contains log notes. Focus on identifying information, plans. Maintain the structure of the notes and present the result in strictly markdown format and use the tables if it's there in the document. Be especially careful with terms .Give the whole content of the document from the header to the footer""",
            llm_instruction="""Structure this log record with attention to:
            CONTENT PRIORITIES:
            - Temporal information (dates, times, durations, intervals)
            - Entry identifiers and sequence numbers
            - Names and identifiers of involved parties
            - Categorical classifications
            - Numerical data and measurements
            - Action items and follow-ups
            - Notes and observations
            - References to other entries

            STRUCTURAL REQUIREMENTS:
            - Maintain strict chronological order
            - Preserve entry relationships
            - Group related information
            - Identify entry hierarchies
            - Link connected entries
            - Maintain column structure
            - Preserve entry boundaries

            OUTPUT FORMAT:
            - Structure as markdown tables
            - Include metadata headers
            - Use consistent date/time formatting
            - Standardize abbreviations
            - Preserve numerical precision
            - Maintain list structures
            - Clear entry separation""",
            preprocessing_defaults={
                'target_dpi': 300,
                'd_bilateral': 9,
                'sigma_color': 75,
                'sigma_space': 75,
                'clip_limit': 2.0,
                'tile_size': 8,
                'alpha_adjust': 1.0,
                'beta_adjust': 10,
                'block_size': 11,
                'C_value': 5
            },
            recommended_models=['llama_parse_gemini_pro', 'llama_parse_gemini_flash', 'llama_parse_gpt4_mini', 'llama_vision', 'pixtral', 'tesseract']
        ),
        'medical_records': DocumentConfig(
            name="Medical Records",
            description="Medical documents, prescriptions, and patient records",
            ocr_instruction="""Please transcribe these hand written Medical Document file for me. Please use markdown formatting. Do not apply line wrapping to any paragraphs. Try to capture headings and subheadings as well as any formatting, such as bold or italic. Omit any text that has been scribbled out and use tabular format if there in the pdf. Try your best to understand the writing and produce a first draft. And also give the full parse of the document. If anything is unclear, follow up with questions at the end. If there try connecting the medical report with the qualifications of the doctor for better understanding of the medical terminologies.""",
            llm_instruction="""Analyze and structure this medical prescription with focus on:

            CRITICAL ELEMENTS:
            - Patient identification details
            - Medication names (generic and brand)
            - Dosage specifications
            - Administration instructions
            - Frequency and duration
            - Prescriber information
            - Date of prescription
            - Refill information

            SAFETY CHECKS:
            - Flag unusual dosages
            - Identify drug interactions
            - Highlight ambiguous instructions

            FORMATTING REQUIREMENTS:
            - Separate sections with clear headers
            - Use standardized medical terminology

            VERIFICATION POINTS:
            - Cross-reference with common drug names
            - Verify prescriber credentials format""",
            preprocessing_defaults={
                'target_dpi': 400,  # Higher DPI for medical documents
                'd_bilateral': 7,
                'sigma_color': 50,
                'sigma_space': 50,
                'clip_limit': 2.5,
                'tile_size': 8,
                'alpha_adjust': 1.2,
                'beta_adjust': 15,
                'block_size': 11,
                'C_value': 4
            },
            recommended_models=['llama_parse_gemini_pro', 'llama_parse_gemini_flash', 'llama_parse_gpt4_mini']
        ),
        'legal_documents': DocumentConfig(
            name="Legal Documents",
            description="Legal contracts, agreements, and court documents",
            ocr_instruction="""This document contains legal notes. Focus on identifying information, plans. Maintain the structure of the notes and present the result in strictly markdown format and use the tables if it's there in the document. Be especially careful with terms .Give the whole content of the document from the header to the footer, and also focus of the legal terms to make sure it's correct.""",
            llm_instruction="""
            Process this legal document with attention to:

            CONTENT IDENTIFICATION:
            - Document type and jurisdiction
            - Party names and roles
            - Defined terms and definitions
            - Key dates and deadlines
            - Monetary values and obligations
            - Conditions and prerequisites
            - Rights and obligations
            - Governing law clauses
            - Signature blocks

            STRUCTURAL ELEMENTS:
            - Section and subsection hierarchy
            - Paragraph numbering
            - Cross-references
            - Schedules and annexures
            - Definitions section
            - Execution blocks
            - Amendment notations
            - Table of contents matching

            FORMATTING REQUIREMENTS:
            - Maintain hierarchical structure
            - Preserve paragraph numbering
            - Retain all legal formatting
            - Keep clause relationships
            - Match original layout
            - Maintain page references
            - Preserve tabular data
            - Indicate any missing sections

            VALIDATION CHECKS:
            - Cross-reference accuracy
            - Definition consistency
            - Party name consistency
            - Date format standardization
            - Numerical value accuracy
            - Section number sequence
            - Missing clause detection
            - Amendment incorporation""",
            preprocessing_defaults={
                'target_dpi': 350,
                'd_bilateral': 9,
                'sigma_color': 60,
                'sigma_space': 60,
                'clip_limit': 1.8,
                'tile_size': 8,
                'alpha_adjust': 1.1,
                'beta_adjust': 5,
                'block_size': 13,
                'C_value': 6
            },
            recommended_models=['llama_parse_gemini_pro', 'llama_parse_gemini_flash', 'llama_parse_gpt4_mini']
        ),
        'financial_records': DocumentConfig(
            name="Financial Records",
            description="Financial statements, invoices, and receipts",
            ocr_instruction="""This document contains financial notes. Focus on identifying information, plans. Maintain the structure of the notes and present the result in strictly markdown format and use the tables if it's there in the document. Be especially careful with terms .Give the whole content of the document from the header to the footer, and also focus of the financial terms to make sure it's correct. Parse the whole document.""",
            llm_instruction="""Analyze and structure this financial document with focus on:

            CRITICAL ELEMENTS:
            - Transaction amounts and totals
            - Account identifiers and references
            - Date and period information
            - Transaction descriptions
            - Balance calculations
            - Currency denominations
            - Payment/Receipt indicators
            - Authorization details

            NUMERICAL PROCESSING:
            - Maintain decimal precision
            - Verify mathematical operations
            - Check running balances
            - Validate totals and subtotals
            - Flag mathematical discrepancies
            - Preserve currency formatting
            - Track cumulative amounts
            - Verify cross-foot totals

            STRUCTURAL REQUIREMENTS:
            - Maintain columnar alignment
            - Preserve table structures
            - Group related transactions
            - Show calculation hierarchy
            - Keep header/footer information
            - Maintain date sequences
            - Preserve account groupings
            - Format as markdown tables

            VALIDATION RULES:
            - Balance accuracy checks
            - Date sequence verification
            - Transaction completeness
            - Mathematical accuracy
            - Currency consistency
            - Account number validation
            - Authorization presence
            - Required field completion""",
            preprocessing_defaults={
                'target_dpi': 300,
                'd_bilateral': 5,
                'sigma_color': 40,
                'sigma_space': 40,
                'clip_limit': 2.2,
                'tile_size': 8,
                'alpha_adjust': 1.0,
                'beta_adjust': 8,
                'block_size': 9,
                'C_value': 4
            },
            recommended_models=['tesseract']
        ),
        'general_config': DocumentConfig(
            name="General OCR",
            description="Overall General handwriting OCR",
            ocr_instruction="""Please transcribe these handwritten notes for me. Please use markdown formatting. Do not apply line wrapping to any paragraphs. Try to capture headings and subheadings as well as any formatting, such as bold or italic. Omit any text that has been scribbled out. Try your best to understand the writing and produce a first draft. If anything is unclear, follow up with questions at the end. Give the output in Markdown strictly. Parse the whole document""",
            llm_instruction="""Process this handwritten document with focus on:

            CONTENT INTERPRETATION:
            - Main text content and flow
            - Numbers and numerical data
            - Personal names and proper nouns
            - Dates and time references
            - Special characters and symbols
            - Abbreviations and shorthand
            - Corrections and strike-throughs
            - Insert marks and additions
            - Underlined or emphasized text

            WRITING STYLE HANDLING:
            - Different handwriting styles
            - Mixed cursive and print
            - Varying text sizes
            - Slant and spacing variations
            - Pressure variations
            - Connected vs disconnected writing
            - Superscript/subscript elements
            - Character overlaps
            - Line spacing irregularities

            DOCUMENT STRUCTURE:
            - Paragraph organization
            - List formats (bullet points, numbers)
            - Indentation patterns
            - Margin notes and annotations
            - Page headers and footers
            - Section breaks and divisions
            - Column arrangements
            - Table structures
            - Diagram labels and captions

            FORMATTING PRESERVATION:
            - Text alignment and justification
            - Line breaks and spacing
            - Paragraph separation
            - Emphasis indicators (underline, circles)
            - Margin spacing
            - Page layout
            - Spatial relationships
            - Drawing elements
            - Diagram positioning

            OUTPUT REQUIREMENTS:
            - Convert to clear markdown format
            - Maintain original structure
            - Preserve emphasis indicators
            - Keep list formatting
            - Retain spatial layout where significant
            - Include relevant annotations
            - Mark unclear or ambiguous text
            - Note alternative interpretations
            - Preserve original organization

            QUALITY VALIDATION:
            - Flag illegible words
            - Mark uncertain interpretations
            - Note context inconsistencies
            - Highlight ambiguous characters
            - Indicate possible alternatives
            - Check logical flow
            - Verify numerical consistency
            - Review name spellings
            - Cross-check dates and references
            """,
            preprocessing_defaults={
                'target_dpi': 300,
                'd_bilateral': 5,
                'sigma_color': 40,
                'sigma_space': 40,
                'clip_limit': 2.2,
                'tile_size': 8,
                'alpha_adjust': 1.0,
                'beta_adjust': 8,
                'block_size': 9,
                'C_value': 4
            },
            recommended_models=['microsoft_trocr', 'llama_parse_gemini_pro', 'llama_parse_gemini_flash', 'llama_parse_gpt4_mini', 'easyocr', 'llama_vision', 'pixtral', 'tesseract']
        )
    }
```

## Usage

The `document_configs.py` module is used to define and retrieve configurations for different document types. These configurations are utilized in the main application to tailor the OCR and LLM processing to specific document types.

### Example Usage

```python
from document_configs import get_document_configs

# Get document configurations
document_configs = get_document_configs()

# Access a specific document configuration
log_notes_config = document_configs['log_notes']

# Print details of the log notes configuration
print(f"Name: {log_notes_config.name}")
print(f"Description: {log_notes_config.description}")
print(f"OCR Instruction: {log_notes_config.ocr_instruction}")
print(f"LLM Instruction: {log_notes_config.llm_instruction}")
print(f"Preprocessing Defaults: {log_notes_config.preprocessing_defaults}")
print(f"Recommended Models: {log_notes_config.recommended_models}")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
