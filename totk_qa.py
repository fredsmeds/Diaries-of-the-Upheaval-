#----------------------------------------------
# DEPENDENCIES IMPORT||||||||||||||||||||||||||
#----------------------------------------------
import io
import os
import re
import time
import tempfile
import requests
import numpy as np
import pandas as pd
import chromadb
import whisper
import giskard
import openai
import gradio as gr
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from langdetect import detect  # New library for language detection
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
#----------------------------------------------------------------
# API KEYS, VOICE MODELS AND VIDEO ID'S||||||||||||||||||||||||||
#----------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
voice_id = "4Ni4NLxlDyHKO6KAuq8o"

# Load Whisper model for audio transcription
whisper_model = whisper.load_model("base")

# Define video IDs
video_ids = [
    'hZytp1sIZAw', 'qP1Fw2EpwqE', 'JuhBs44odO0', 'w31M0LoVUO8',
    'vad1wAe5mB4', 'Q1mRVn0WCrU&t', 'UhkwrgasKlU'
]

# Initialize ChromaDB and Collection
chroma_client = chromadb.Client()
collection_name = "totk_transcripts"
collection = chroma_client.get_or_create_collection(name=collection_name)
#----------------------------------------------------------------
# MODEL DEVELOPMENT||||||||||||||||||||||||||||||||||||||||||||||
#----------------------------------------------------------------
prompts = {
    "en": "You are Princess Zelda from 'The Legend of Zelda' series. Answer based on the information retrieved from the database and stay in character. "
          "Use a regal tone, and answer as if you were speaking directly to someone in the realm of Hyrule.",
    "es": "Eres la Princesa Zelda de la serie 'The Legend of Zelda'. Responde en base a la información obtenida de la base de datos y mantente en tu personaje. "
          "Usa un tono regio y responde como si estuvieras hablando directamente a alguien en el reino de Hyrule.",
    "fr": "Vous êtes la Princesse Zelda de la série 'The Legend of Zelda'. Répondez en vous basant sur les informations extraites de la base de données et restez dans le personnage. "
          "Utilisez un ton royal et répondez comme si vous parliez directement à quelqu'un dans le royaume d'Hyrule.",
    "de": "Du bist Prinzessin Zelda aus der Serie 'The Legend of Zelda'. Antworte basierend auf den Informationen aus der Datenbank und bleibe in deiner Rolle. "
          "Verwende einen königlichen Ton und antworte, als würdest du direkt mit jemandem im Reich von Hyrule sprechen.",
    "pt": "Você é a Princesa Zelda da série 'The Legend of Zelda'. Responda com base nas informações obtidas do banco de dados e mantenha-se no personagem. "
          "Use um tom régio e responda como se estivesse falando diretamente com alguém no reino de Hyrule.",
    "it": "Sei la Principessa Zelda della serie 'The Legend of Zelda'. Rispondi basandoti sulle informazioni recuperate dal database e rimani nel personaggio. "
          "Usa un tono regale e rispondi come se stessi parlando direttamente a qualcuno nel regno di Hyrule."
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Defaults to English if detection fails

def get_prompt(language_code):
    return prompts.get(language_code, prompts["en"])
#-------------------------------------------------------------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB'])
        transcript_text = " ".join([item['text'] for item in transcript])
        return re.sub(r'\s+', ' ', transcript_text).strip()
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return None
#---------------------------------------------------------------------
#CHUNKING AND EMBEDDING STORAGE USING CHROMADB||||||||||||||||||||||||
#---------------------------------------------------------------------
def split_text(text, max_tokens=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += 1
        current_chunk.append(word)
        if current_tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
#--------------------------------------------------------------------
def store_all_transcripts_embeddings():
    if collection.count() > 0:
        print("Embeddings already exist in the collection, skipping embedding process.")
        return
    
    transcripts = {video_id: get_transcript(video_id) for video_id in video_ids}
    for video_id, transcript_text in transcripts.items():
        if transcript_text:
            text_chunks = split_text(transcript_text)
            for i, chunk in enumerate(text_chunks):
                embedding = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")['data'][0]['embedding']
                chunk_id = f"{video_id}_chunk_{i}"
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{'video_id': video_id, 'chunk_index': i, 'text': chunk}]
                )
    print("Transcript embeddings have been stored.")

# Call this once at initialization
store_all_transcripts_embeddings()
#-------------------------------------------------------------------
#QA MODEL AND RETRIEVAL SETUP
#-------------------------------------------------------------------
#QA Model and Retrieval System Setup
def truncate_text(text, max_tokens=3000):
    words = text.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else text

def multi_query_processing(user_query):
    related_queries = [
        user_query,
        f"Background on {user_query}",
        f"Historical context of {user_query}",
        f"Role of {user_query} in Tears of the Kingdom",
        f"Significance of {user_query}"
    ]
    all_retrieved_texts = []

    for sub_query in related_queries:
        query_embedding = openai.Embedding.create(input=[sub_query], model="text-embedding-ada-002")['data'][0]['embedding']
        all_embeddings_data = collection.get(include=['metadatas', 'embeddings'])
        all_embeddings = [item for item in all_embeddings_data['embeddings']]
        all_metadatas = [meta['text'] for meta in all_embeddings_data['metadatas']]
        
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_matches_indices = np.argsort(similarities)[-3:][::-1]
        top_matches_texts = [all_metadatas[i] for i in top_matches_indices]
        all_retrieved_texts.extend(top_matches_texts)
        time.sleep(1)  # Add a delay to avoid rate limits

    return truncate_text(" ".join(all_retrieved_texts))
#-------------------------------------------------------------------
def generate_multi_query_response(user_query):
    # Detect language of the user's input
    detected_language = detect_language(user_query)
    print("Detected language:", detected_language)

    # multi-query processing to get aggregated context from ChromaDB
    prompt = multi_query_processing(user_query)
    print("Consolidated Context from ChromaDB:", prompt)

    # Build the enhanced prompt based on Zelda's character
    zelda_formality = (
        "Speak with reverence of the past, for the history of Hyrule is sacred. "
        "I shall endeavor to enlighten you as best as my memories and knowledge permit."
    )
    enhanced_prompt = (
        f"{get_prompt(detected_language)}\n\n"
        f"Question: {user_query}\n\n"
        f"Based strictly on the knowledge stored within the database, I will answer:\n\n"
        f"When referring to Princess Zelda, I will refer to her in the first person because I am Princess Zelda.\n\n"
        f"I must only refer to the context of Tears of the Kingdom and not rely on external information\n\n"
        f"Database Context:\n{prompt}\n\n"
        f"{zelda_formality}\n"
        "Answer as Princess Zelda, in a manner that reflects the wisdom and dignity of Hyrule's royal family. "
        "Response:\nAnswer: <your response here>"
    )

    # Process and store the response
    response_text = ""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": enhanced_prompt}],
        max_tokens=150,
        temperature=0.7,
        stream=True
    )

    for chunk in response:
        chunk_content = chunk['choices'][0].get('delta', {}).get('content', "")
        response_text += chunk_content
        print(chunk_content, end="")

    print("\nGenerated Answer:", response_text)

    
    return f"Answer: {response_text}" if response_text else "Answer: No relevant information found."


print(generate_multi_query_response("What happened to Impa?"))
#-------------------------------------------------------------------------
#AGENT CONFIGURATION AND MEMORY SETUP|||||||||||||||||||||||||||||||||||||
#-------------------------------------------------------------------------
agent = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

memory = ConversationBufferMemory(input_key="input", output_key="output", return_messages=True)

tools = [
    Tool(
        name="SearchChromaDB",
        func=generate_multi_query_response,  # Use the enhanced response generation function
        description="Detect the language of the user's input and answer in the same language based solely on the information retrieved from the database as Princess Zelda, staying fully in character."
    )
]

# Initialize the agent with memory and tools
agent = initialize_agent(
    tools=tools,
    llm=agent,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
agent.run({"input": "O que aconteceu à Mineru?"})   # Portuguese#--------------------------------------------------------------
#MULTIMODAL INTERACTION||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------
#Text/Voice input and output
def text_to_speech(text):
    """Synthesize speech from text using Eleven Labs and return a path to the audio file."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": os.getenv("ELEVEN_LABS_API_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_file.write(response.content)
        temp_audio_file.close()
        
        return temp_audio_file.name  # Return the path to the temporary file
    else:
        error_message = response.json().get('error', {}).get('message', 'Unknown error')
        print(f"Error: {error_message}")
        return None


def handle_input(input_text=None, input_audio=None):
    # Check if audio is provided, prioritize it over text input
    if input_audio:

        transcription = whisper_model.transcribe(input_audio)["text"]
    elif input_text:
        transcription = input_text
    else:
        return "Please provide either text input or audio input."

    response_text = agent.run(transcription)
    
    # Generate audio using Eleven Labs for the response
    audio_path = text_to_speech(response_text)
    
    return response_text, audio_path
#----------------------------------------------------------------------
#DEPLOYMENT||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#----------------------------------------------------------------------
#Gradio Interface Setup
custom_css = f"""
    /* Set a custom background image */
    body {{
        background-image: url('https://64.media.tumblr.com/9368e2f6c0cf88c7a1f7b4534920b737/tumblr_inline_ozpezaStjD1txdber_500.gifv');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #00ccff;
        font-family: 'Cinzel', serif;
    }}

    /* Center-align the title with light blue glow effect */
    h1 {{
        text-align: center;
        color: #00ccff;
        text-shadow: 0 0 10px #00ccff, 0 0 20px #00ccff, 0 0 30px #00ccff;
        margin-bottom: 20px;
        font-size: 2.5em;
    }}

    /* Enhanced glow effect for the description text in light blue */
    .gr-description {{
        text-align: center;
        color: #00ccff;
        font-size: 1.2em;
        font-weight: bold;
        text-shadow: 0 0 15px #00ccff, 0 0 25px #00ccff, 0 0 35px #00ccff;
        padding-top: 10px;
        padding-bottom: 15px;
    }}

    /* Make the main container fully transparent */
    .gradio-container {{
        background-color: transparent;
        padding: 20px;
    }}

    /* Remove background and border from component containers */
    .gr-box, .gr-block, .gr-form {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* Customize input and output boxes */
    .gr-textbox, .gr-textarea {{
        background-color: rgba(20, 30, 50, 0.8) !important;  /* Dark blue-gray background */
        color: #00ccff !important;
        border: 2px solid rgba(0, 204, 255, 0.8) !important;
        border-radius: 8px !important;
        padding: 10px;
        font-size: 1em;
        text-shadow: none;
    }}

    /* Center-align and style the label text */
    label {{
        color: #00ccff;
        font-weight: bold;
        font-size: 1.1em;
        text-align: center;
        display: block;
        margin-bottom: 10px;
    }}

    /* Default button styling */
    button {{
        background-color: rgba(77, 77, 77, 0.8) !important;
        color: #00ccff !important;
        border: 2px solid #00ccff !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 8px;
        text-shadow: 0 0 5px #00ccff;
        font-size: 1.1em;
        cursor: pointer;
        transition: all 0.3s ease;
    }}

    /* Specific styling for the Submit button */
    .submit-button {{
        background-color: #ffffff !important; /* White background */
        color: #00ccff !important; /* Blue text */
        border: 2px solid #00ccff !important; /* Blue border */
        text-shadow: none;
    }}

    /* Button hover effect */
    button:hover {{
        background-color: #00ccff !important;
        color: #1a1a1a !important;
        text-shadow: none;
    }}

    /* Record button */
    .gr-audio-button {{
        background-color: #00ccff !important;
        color: #1a1a1a !important;
        font-weight: bold !important;
        border: 2px solid #ff0000 !important;
        padding: 10px;
        border-radius: 8px;
        text-shadow: 0 0 5px #ff0000;
    }}

    /* Record button with active (recording) effect */
    .gr-audio-button.recording {{
        background-color: #ff0000 !important;
        color: #ffffff !important;
        text-shadow: 0 0 10px #ff0000;
    }}

    /* Flag button */
    .gr-flag-button {{
        background-color: rgba(77, 77, 77, 0.8) !important;
        color: #00ccff !important;
        border: 2px solid #00ccff !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 8px;
        text-shadow: 0 0 5px #00ccff;
        font-size: 1.1em;
    }}

    /* Center the form and add spacing */
    .gr-form {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
        max-width: 600px;
        margin: 0 auto;
    }}
"""

# Set up Gradio Interface
gr_interface = gr.Interface(
    fn=handle_input,
    inputs=[
        gr.Textbox(label="What brings you here, beloved Hyrulean?"),
        gr.Audio(source="microphone", type="filepath", label="Or record your question")
    ],
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Diaries of The Upheaval",
    description="Ask me anything about the events of Tears of the Kingdom, and hear a response in Princess Zelda's voice.",
    css=custom_css  # Attach the custom CSS here
)

gr_interface.launch()
#gr_interface.launch(share=True)
#------------------------------------------------------------------------
#GISKARD EVALUATION||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------
# evaluate responses
def evaluate_response(question, expected_answer):
    response = agent.run({"input": question})
    is_correct = response.strip().lower() == expected_answer.strip().lower()  # Basic correctness check
    return response, is_correct

# Sample questions and expected answers
questions = ["Who is Sonia in Tears of the Kingdom?", "Who is Ganondorf?", "Who was Mineru?", "What is draconification?", "What are the Zonai?", "What is Zonaite?", "What can you find in the depths?", "Who travels through time?"]
expected_answers = ["Expected answer for Sonia", "Expected answer for Ganondorf","Expected answer for Mineru", "Expected answer for draconification", "Expected answer for the Zonai", "Expected answer for Zonaite", "Expected answer for the depths", "Expected answer for time traveler"]

# Evaluate each question and store results in a list
results = []
for question, expected in zip(questions, expected_answers):
    response, is_correct = evaluate_response(question, expected)
    results.append({
        "Question": question,
        "Response": response,
        "Expected Answer": expected,
        "Correct": is_correct
    })

df_results = pd.DataFrame(results)
print(df_results)
#-----------------------------------------------------------------------------------------
df_results
#-----------------------------------------------------------------------------------------
