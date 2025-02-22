import whisper
from pydub import AudioSegment
import csv
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import google.generativeai as genai
import warnings
import os
from dotenv import load_dotenv

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Whisper model
model = whisper.load_model("base")  # Can be "tiny" or "large" based on accuracy/speed

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="transcriptions")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def transcribe_with_whisper(file_path):
    """Convert audio to text using Whisper."""
    
    # Convert to WAV if input is M4A
    if file_path.endswith(".m4a"):
        audio = AudioSegment.from_file(file_path, format="m4a")
        temp_file_path = "temp.wav"
        audio.export(temp_file_path, format="wav")
        file_path = temp_file_path
    
    # Perform transcription
    result = model.transcribe(file_path)
    transcription = result["text"]
    
    return transcription

def save_transcription_to_csv(transcription, output_file="transcription.csv"):
    """Save transcription as CSV."""
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Transcription"])  # Column header
        writer.writerow([transcription])    # Transcription content

def store_in_vectordb(text, file_id):
    """Convert transcription to embedding and store in ChromaDB."""
    embedding = genai.embed_content(model="models/embedding-001", content=text)
    
    collection.add(
        ids=[file_id],  # Unique identifier (e.g., filename)
        embeddings=[embedding["embedding"]],
        metadatas=[{"text": text}]
    )

def retrieve_past_context(user_query):
    """Retrieve similar past transcriptions from ChromaDB."""
    query_embedding = genai.embed_content(model="models/embedding-001", content=user_query)
    
    # Fetch top 3 relevant past conversations
    results = collection.query(query_embeddings=[query_embedding["embedding"]], n_results=3)
    
    past_contexts = [r["text"] for r in results["metadatas"][0]] if results["metadatas"] else []
    
    return past_contexts

def chat_with_memory(user_query):
    """Use past context and Gemini for chatbot response, including possible causes and suggested medicines."""
    past_contexts = retrieve_past_context(user_query)
    
    # Enhanced medical prompt
    prompt = f"""
    You are a highly knowledgeable virtual medical assistant. Analyze the user's symptoms carefully and provide:
    
    1. **Possible Causes** - List potential reasons for the symptoms.
    2. **Recommended Medicines** - Suggest common over-the-counter or prescribed medications for symptom relief.
    3. **Precautions & Next Steps** - Advise on whether medical consultation is necessary and any home remedies.

    Previous Medical Context (if available): {past_contexts}
    
    **User Symptoms:** {user_query}
    
    **AI Response:**
    """

    # Use Gemini-Pro for generating response
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    
    return response.text if response else "I'm sorry, I couldn't generate a response."



# ===== RUN TRANSCRIPTION & STORE IN VECTOR DB =====
file_path = "New recording 1.m4a"  # Replace with your file
transcript = transcribe_with_whisper(file_path)
print("Transcription:", transcript)

# Save transcription locally as CSV
save_transcription_to_csv(transcript)

# Store in VectorDB
store_in_vectordb(transcript, file_id=file_path)

print("Transcription stored in VectorDB ✅")

# ===== USE TRANSCRIPT AS USER QUERY =====
response = chat_with_memory(transcript)
print("Chatbot Response:", response)
