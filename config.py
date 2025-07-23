import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = "upfl-ar-final-2024-ocr.pdf"
FAISS_INDEX = "faiss_index"