import streamlit as st
from PIL import Image
import nest_asyncio 
import logging
import cv2
import numpy as np
from llm import restructure_response, regen_model_types, regen_document_types
from ocr_models import get_available_models, process_image_with_model
from document_configs import get_document_configs

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()

class OCRApp:
    def __init__(self):
        self.setup_page_config()
        try:
            load_dotenv()
            logger.info("Environment variable loaded successfully")            
        except Exception as e:
            logger.error(f"Error setting API keys: {str(e)}")
        self.document_configs = get_document_configs()
        self.initialize_session_state()
        
    def setup_page_config(self):
        import os
        import sys
        
        st.set_page_config(page_title="Document Parser", page_icon="📄", layout="wide")
        
        # Environment detection
        is_colab = 'google.colab' in sys.modules
        is_streamlit_cloud = os.getenv('STREAMLIT_CLOUD') == 'true'
        
        # Define CSS paths for different environments
        if is_colab:
            css_path = "/content/OCR_Accelerator/assets/styless.css"
        elif is_streamlit_cloud:
            css_path = "path/for/streamlit/cloud/styless.css"
        else:
            css_path = "assets/styless.css"
        
        # CSS loading with multiple fallback options
        css_loaded = False
        
        # Try loading custom CSS
        try:
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                css_loaded = True
        except FileNotFoundError:
            print(f"Primary CSS file not found at {css_path}")
        
        # If primary CSS fails, try loading fallback CSS
        if not css_loaded:
            fallback_paths = [
                "fallback/styless.css",
                "alternative/path/styless.css"
            ]
            
            for path in fallback_paths:
                try:
                    with open(path) as f:
                        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                        css_loaded = True
                        break
                except FileNotFoundError:
                    continue
        
        # If all CSS loading attempts fail, use inline CSS
        if not css_loaded:
            default_css = """
            /* Default CSS styles */
            .stApp {
                background-color: #f0f2f6;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
            }
            /* Add more default styles as needed */
            """
            st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

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
            
        # Add new session state for AI conclusion toggle
        if 'show_ai_conclusion' not in st.session_state:
            st.session_state.show_ai_conclusion = False
            
        if 'llm_processed_result' not in st.session_state:
            st.session_state.llm_processed_result = None
            
    def render_document_type_selection(self):
        st.sidebar.header("Document Type")
        
        selected_type = st.sidebar.selectbox(
            "Select Document Type",
            options=list(self.document_configs.keys()),
            format_func=lambda x: self.document_configs[x].name
        )
        
        if selected_type != st.session_state.document_type:
            st.session_state.document_type = selected_type
            # Update preprocessing params to defaults for this document type
            if st.session_state.preprocessing_enabled:
                st.session_state.preprocessing_params = self.document_configs[selected_type].preprocessing_defaults.copy()
            st.session_state.selected_model = None  # Reset model selection
            
        if selected_type:
            st.sidebar.markdown(f"*{self.document_configs[selected_type].description}*")
            
            # Show recommended models
            st.sidebar.markdown("**Recommended Models:**")
            for model in self.document_configs[selected_type].recommended_models:
                model_config = get_available_models()[model]
                st.sidebar.markdown(f"- {model_config.name}")

    def render_model_selection(self):
        if not st.session_state.document_type:
            st.sidebar.warning("Please select a document type first")
            return

        st.sidebar.header("Model Selection")
        available_models = get_available_models()
        doc_config = self.document_configs[st.session_state.document_type]
        
        # Filter to show only recommended models for this document type
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

    def process_image(self):
        if not st.session_state.selected_model or not st.session_state.document_type:
            return
        
        uploaded_image = st.session_state.uploaded_image
        if uploaded_image:
            with st.spinner("Processing image..."):
                try:
                    image = Image.open(uploaded_image)
                    image = self.preprocess_image(image)
                    
                    # Get document-specific instructions
                    doc_config = self.document_configs[st.session_state.document_type]
                    
                    # Process with selected model and document-specific configuration
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


    def show_preprocessing_controls(self):
        if not st.session_state.preprocessing_enabled:
            return

        st.sidebar.header("Preprocessing Controls")
        
        # Add reset to defaults button
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

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing steps to the image based on current parameters
        """
        if not st.session_state.preprocessing_enabled:
            return image

        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        params = st.session_state.preprocessing_params
        
        # Apply preprocessing steps
        # Bilateral Filter
        img_array = cv2.bilateralFilter(
            img_array,
            params['d_bilateral'],
            params['sigma_color'],
            params['sigma_space']
        )
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
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

        # Brightness and Contrast adjustment
        img_array = cv2.convertScaleAbs(
            img_array,
            alpha=params['alpha_adjust'],
            beta=params['beta_adjust']
        )

        # Convert back to PIL Image
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_array)

    def render_processing_controls(self):
        st.sidebar.header("Processing Controls")
        
        # Add AI conclusion toggle
        st.session_state.show_ai_conclusion = st.sidebar.checkbox(
            "Generate AI Conclusion",
            value=st.session_state.show_ai_conclusion,
            help="Enable to get an AI-generated analysis of the processed document"
        )
        
        if st.session_state.selected_model:
            st.sidebar.button("Process Image", on_click=self.process_image)
        else:
            st.sidebar.info("Select a model to process the image")

    def check_regeneration(self):
        if st.session_state.selected_model in regen_model_types and st.session_state.document_type in regen_document_types:
            return True
        else:
            return False
            

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
                
                # Only process with LLM if AI conclusion is enabled and regeneration is applicable
                if st.session_state.show_ai_conclusion and self.check_regeneration():
                    with st.spinner("Generating AI conclusion... 🤖"):
                        st.session_state.llm_processed_result = restructure_response(
                            result, 
                            self.document_configs[st.session_state.document_type].llm_instruction
                        )
                st.success("Image processed successfully!")

    def download_results(self, result: str, file_extension: str):
        """Function to handle downloading results."""
        # Define the filename based on the provided extension
        filename = f"results.{file_extension}"
        
        # Create a download button
        st.download_button(
            label="Download Results",
            data=result,
            file_name=filename,
            mime=f"text/{file_extension}" if file_extension == "md" else f"application/octet-stream",
            key="download_button"
        )
            

    def render_processing_result(self):
        if st.session_state.processing_result:
            st.header("Processing Result")
            st.write(st.session_state.processing_result)
            
            # Only show AI conclusion if enabled and available
            if st.session_state.show_ai_conclusion and st.session_state.llm_processed_result:
                st.header("AI Conclusion")
                st.markdown(st.session_state.llm_processed_result)
            
            download_format = st.selectbox("Choose download format:", ["Markdown", "Text"])
            file_extension = "md" if download_format == "Markdown" else "txt"

            # Create a download button
            download_button = st.button("Download Results")
            if download_button:
                result_text = st.session_state.processing_result
                if st.session_state.show_ai_conclusion and st.session_state.llm_processed_result:
                    result_text += "\n\n## AI Conclusion\n\n" + st.session_state.llm_processed_result
                self.download_results(result_text, file_extension)


    def run(self):
        st.title("Document Parser")
        
        # Document type selection
        self.render_document_type_selection()
        
        # Model selection
        self.render_model_selection()
        
        # AI Conclusion
        st.sidebar.header("AI Conclusion")
        st.session_state.show_ai_conclusion = st.sidebar.checkbox(
            "Enable AI Conclusions",
            value=st.session_state.show_ai_conclusion
        )
        
        # Preprocessing toggle and controls
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
        
        # Image upload and processing
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["jpg", "jpeg", "png", "bmp", "tiff"]
        )
        
        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            
            # Display original and processed images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                processed_image = self.preprocess_image(Image.open(uploaded_file))
                st.image(processed_image, use_column_width=True)
            
            # Process button
            if st.button("Process Document"):
                self.process_image()
        
        # Display results
        self.render_processing_result()
        
        # Application information
        # self.render_info()
if __name__ == '__main__':
    app = OCRApp()
    app.run()
