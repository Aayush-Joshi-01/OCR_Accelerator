import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
logger.info("Setting up Gemini API key")
# os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
if not os.environ["GEMINI_API_KEY"]:
    logger.warning("Gemini API key is empty")
else:
    logger.info("Gemini API key set successfully")


regen_document_types = ["medical_records", "log_notes"]
regen_model_types = ["llama_parse_gemini_pro", "llama_parse_gemini_flash", "llama_parse_gpt4_mini"]

def restructure_response(result, llm_instruction):
    logger.info("Starting restructure_response function")
    
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        logger.info("Gemini API configured")
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {str(e)}")
        raise

    generation_config = {
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    logger.info("Generation config set")

    try:
        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction = llm_instruction
        )
        logger.info("Generative model initialized")
    except Exception as e:
        logger.error(f"Error initializing generative model: {str(e)}")
        raise
        logger.info("Chat session started")
    except Exception as e:
        logger.error(f"Error starting chat session: {str(e)}")
        raise

    try:
        response = model.generate_content(result)
        logger.info("Message sent and response received")
        return response.text
    except Exception as e:
        logger.error(f"Error sending message or receiving response: {str(e)}")
        raise

logger.info("restructure_response function defined successfully")