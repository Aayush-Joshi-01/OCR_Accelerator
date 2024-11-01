
# OCR Accelerator - `app.py`

## Overview

The `app.py` module is the main application file for the OCR Accelerator project. It integrates all components and provides the user interface using Streamlit. This module manages the application state, handles user interactions, and orchestrates the document processing workflow.

## Features

- **Document Type Selection**: Allows users to select the type of document they want to process.
- **Model Selection**: Provides a UI for selecting an OCR model from the available options.
- **Preprocessing Controls**: Offers controls to adjust image preprocessing parameters for enhanced OCR accuracy.
- **AI Conclusion**: Generates AI-based conclusions for processed documents if enabled.
- **Download Results**: Allows users to download processed results in Markdown or Text format.

## Key Components

### `OCRApp` Class

The `OCRApp` class manages the application state and integrates various functionalities.

#### Methods

- **`__init__`**: Initializes the application, loads environment variables, and sets up the session state.
- **`setup_page_config`**: Configures the Streamlit page settings.
- **`initialize_session_state`**: Initializes the session state for various parameters.
- **`render_document_type_selection`**: Renders the document type selection UI.
- **`render_model_selection`**: Renders the model selection UI.
- **`process_image`**: Processes the uploaded image using the selected OCR model.
- **`show_preprocessing_controls`**: Displays preprocessing controls for image enhancement.
- **`preprocess_image`**: Applies preprocessing steps to the image.
- **`render_processing_controls`**: Renders the processing controls UI.
- **`check_regeneration`**: Checks if the selected model and document type require regeneration.
- **`process_image`**: Processes the image and optionally generates AI conclusions.
- **`download_results`**: Handles downloading the processed results.
- **`render_processing_result`**: Renders the processed results and AI conclusions.
- **`run`**: Main function to run the application.

### Detailed Method Descriptions

#### `__init__`

Initializes the application, loads environment variables, and sets up the session state.

```python
def __init__(self):
    self.setup_page_config()
    try:
        load_dotenv()
        logger.info("Environment variable loaded successfully")
    except Exception as e:
        logger.error(f"Error setting API keys: {str(e)}")
    self.document_configs = get_document_configs()
    self.initialize_session_state()
```

#### `setup_page_config`

Configures the Streamlit page settings.

```python
def setup_page_config(self):
    st.set_page_config(page_title="Document Parser", page_icon="📄", layout="wide")
    try:
        with open("assets/styless.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
```

#### `initialize_session_state`

Initializes the session state for various parameters.

```python
def initialize_session_state(self):
    if 'preprocessing_enabled' not in st.session_state:
        st.session_state.preprocessing_enabled = False
    if 'preprocessing_params' not in st.session_state:
        st.session_state.preprocessing_params = self.document_configs['log_notes'].preprocessing_defaults.copy()
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'document_type' not in st.session_state:
        st.session_state.document_type = None
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None
    if 'show_ai_conclusion' not in st.session_state:
        st.session_state.show_ai_conclusion = False
    if 'llm_processed_result' not in st.session_state:
        st.session_state.llm_processed_result = None
```

#### `render_document_type_selection`

Renders the document type selection UI.

```python
def render_document_type_selection(self):
    st.sidebar.header("Document Type")
    selected_type = st.sidebar.selectbox(
        "Select Document Type",
        options=list(self.document_configs.keys()),
        format_func=lambda x: self.document_configs[x].name
    )
    if selected_type != st.session_state.document_type:
        st.session_state.document_type = selected_type
        if st.session_state.preprocessing_enabled:
            st.session_state.preprocessing_params = self.document_configs[selected_type].preprocessing_defaults.copy()
        st.session_state.selected_model = None
    if selected_type:
        st.sidebar.markdown(f"*{self.document_configs[selected_type].description}*")
        st.sidebar.markdown("**Recommended Models:**")
        for model in self.document_configs[selected_type].recommended_models:
            model_config = get_available_models()[model]
            st.sidebar.markdown(f"- {model_config.name}")
```

#### `render_model_selection`

Renders the model selection UI.

```python
def render_model_selection(self):
    if not st.session_state.document_type:
        st.sidebar.warning("Please select a document type first")
        return
    st.sidebar.header("Model Selection")
    available_models = get_available_models()
    doc_config = self.document_configs[st.session_state.document_type]
    recommended_models = doc_config.recommended_models
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: available_models[x].name
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.processing_result = None
    if selected_model:
        st.sidebar.markdown(f"*{available_models[selected_model].description}*")
```

#### `process_image`

Processes the uploaded image using the selected OCR model.

```python
def process_image(self):
    if not st.session_state.selected_model or not st.session_state.document_type:
        return
    uploaded_image = st.session_state.uploaded_image
    if uploaded_image:
        with st.spinner("Processing image..."):
            try:
                image = Image.open(uploaded_image)
                image = self.preprocess_image(image)
                doc_config = self.document_configs[st.session_state.document_type]
                result = process_image_with_model(
                    image=image,
                    model_key=st.session_state.selected_model,
                    document_type=st.session_state.document_type,
                    ocr_instruction=doc_config.ocr_instruction,
                    llm_instruction=doc_config.llm_instruction
                )
                st.session_state.processing_result = result
                st.success(f"{doc_config.name} processed successfully!")
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
```

#### `show_preprocessing_controls`

Displays preprocessing controls for image enhancement.

```python
def show_preprocessing_controls(self):
    if not st.session_state.preprocessing_enabled:
        return
    st.sidebar.header("Preprocessing Controls")
    if st.sidebar.button("Reset to Document Defaults"):
        if st.session_state.document_type:
            st.session_state.preprocessing_params = self.document_configs[
                st.session_state.document_type
            ].preprocessing_defaults.copy()
            st.experimental_rerun()
    tabs = st.sidebar.tabs(["Basic", "Filter", "Enhancement", "Threshold"])
    with tabs[0]:
        st.session_state.preprocessing_params['target_dpi'] = st.slider(
            "Target DPI", min_value=100, max_value=600,
            value=st.session_state.preprocessing_params['target_dpi'], step=50)
    with tabs[1]:
        st.session_state.preprocessing_params['d_bilateral'] = st.slider(
            "Bilateral Filter Size", min_value=5, max_value=15,
            value=st.session_state.preprocessing_params['d_bilateral'], step=2)
        st.session_state.preprocessing_params['sigma_color'] = st.slider(
            "Sigma Color", min_value=25, max_value=150,
            value=st.session_state.preprocessing_params['sigma_color'], step=25)
        st.session_state.preprocessing_params['sigma_space'] = st.slider(
            "Sigma Space", min_value=25, max_value=150,
            value=st.session_state.preprocessing_params['sigma_space'], step=25)
    with tabs[2]:
        st.session_state.preprocessing_params['clip_limit'] = st.slider(
            "CLAHE Clip Limit", min_value=0.5, max_value=5.0,
            value=st.session_state.preprocessing_params['clip_limit'], step=0.5)
        st.session_state.preprocessing_params['tile_size'] = st.slider(
            "CLAHE Tile Size", min_value=2, max_value=16,
            value=st.session_state.preprocessing_params['tile_size'], step=2)
        st.session_state.preprocessing_params['alpha_adjust'] = st.slider(
            "Contrast", min_value=0.5, max_value=3.0,
            value=st.session_state.preprocessing_params['alpha_adjust'], step=0.1)
        st.session_state.preprocessing_params['beta_adjust'] = st.slider(
            "Brightness", min_value=-50, max_value=50,
            value=st.session_state.preprocessing_params['beta_adjust'], step=5)
    with tabs[3]:
        st.session_state.preprocessing_params['block_size'] = st.slider(
            "Block Size", min_value=3, max_value=51,
            value=st.session_state.preprocessing_params['block_size'], step=2)
        st.session_state.preprocessing_params['C_value'] = st.slider(
            "C Value", min_value=1, max_value=20,
            value=st.session_state.preprocessing_params['C_value'], step=1)
```

#### `preprocess_image`

Applies preprocessing steps to the image.

```python
def preprocess_image(self, image: Image.Image) -> Image.Image:
    if not st.session_state.preprocessing_enabled:
        return image
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    params = st.session_state.preprocessing_params
    img_array = cv2.bilateralFilter(
        img_array,
        params['d_bilateral'],
        params['sigma_color'],
        params['sigma_space']
    )
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=params['clip_limit'],
            tileGridSize=(params['tile_size'], params['tile_size'])
        )
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(
            clipLimit=params['clip_limit'],
            tileGridSize=(params['tile_size'], params['tile_size'])
        )
        img_array = clahe.apply(img_array)
    img_array = cv2.convertScaleAbs(
        img_array,
        alpha=params['alpha_adjust'],
        beta=params['beta_adjust']
    )
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_array)
```

#### `render_processing_controls`

Renders the processing controls UI.

```python
def render_processing_controls(self):
    st.sidebar.header("Processing Controls")
    st.session_state.show_ai_conclusion = st.sidebar.checkbox(
        "Generate AI Conclusion",
        value=st.session_state.show_ai_conclusion,
        help="Enable to get an AI-generated analysis of the processed document"
    )
    if st.session_state.selected_model:
        st.sidebar.button("Process Image", on_click=self.process_image)
    else:
        st.sidebar.info("Select a model to process the image")
```

#### `check_regeneration`

Checks if the selected model and document type require regeneration.

```python
def check_regeneration(self):
    if st.session_state.selected_model in regen_model_types and st.session_state.document_type in regen_document_types:
        return True
    else:
        return False
```

#### `process_image`

Processes the image and optionally generates AI conclusions.

```python
def process_image(self):
    if not st.session_state.selected_model or not st.session_state.document_type:
        return
    uploaded_image = st.session_state.uploaded_image
    if uploaded_image:
        with st.spinner("Processing image...📜"):
            image = Image.open(uploaded_image)
            image = self.preprocess_image(image)
            result = process_image_with_model(image, st.session_state.selected_model, st.session_state.document_type)
            st.session_state.processing_result = result
            if st.session_state.show_ai_conclusion and self.check_regeneration():
                with st.spinner("Generating AI conclusion... 🤖"):
                    st.session_state.llm_processed_result = restructure_response(
                        result,
                        self.document_configs[st.session_state.document_type].llm_instruction
                    )
            st.success("Image processed successfully!")
```

#### `download_results`

Handles downloading the processed results.

```python
def download_results(self, result: str, file_extension: str):
    filename = f"results.{file_extension}"
    st.download_button(
        label="Download Results",
        data=result,
        file_name=filename,
        mime=f"text/{file_extension}" if file_extension == "md" else f"application/octet-stream",
        key="download_button"
    )
```

#### `render_processing_result`

Renders the processed results and AI conclusions.

```python
def render_processing_result(self):
    if st.session_state.processing_result:
        st.header("Processing Result")
        st.write(st.session_state.processing_result)
        if st.session_state.show_ai_conclusion and st.session_state.llm_processed_result:
            st.header("AI Conclusion")
            st.markdown(st.session_state.llm_processed_result)
        download_format = st.selectbox("Choose download format:", ["Markdown", "Text"])
        file_extension = "md" if download_format == "Markdown" else "txt"
        download_button = st.button("Download Results")
        if download_button:
            result_text = st.session_state.processing_result
            if st.session_state.show_ai_conclusion and st.session_state.llm_processed_result:
                result_text += "\n\n## AI Conclusion\n\n" + st.session_state.llm_processed_result
            self.download_results(result_text, file_extension)
```

#### `run`

Main function to run the application.

```python
def run(self):
    st.title("Document Parser")
    self.render_document_type_selection()
    self.render_model_selection()
    st.sidebar.header("AI Conclusion")
    st.session_state.show_ai_conclusion = st.sidebar.checkbox(
        "Enable AI Conclusions",
        value=st.session_state.show_ai_conclusion
    )
    st.sidebar.header("Preprocessing Options")
    preprocessing_enabled = st.sidebar.checkbox(
        "Enable Image Preprocessing",
        value=st.session_state.preprocessing_enabled
    )
    if preprocessing_enabled != st.session_state.preprocessing_enabled:
        st.session_state.preprocessing_enabled = preprocessing_enabled
        if preprocessing_enabled and st.session_state.document_type:
            st.session_state.preprocessing_params = self.document_configs[
                st.session_state.document_type
            ].preprocessing_defaults.copy()
    if preprocessing_enabled:
        self.show_preprocessing_controls()
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_column_width=True)
        with col2:
            st.subheader("Processed Image")
            processed_image = self.preprocess_image(Image.open(uploaded_file))
            st.image(processed_image, use_column_width=True)
        if st.button("Process Document"):
            self.process_image()
    self.render_processing_result()
```

## Usage

To run the OCR Accelerator application, execute the following command:
```bash
streamlit run app.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
