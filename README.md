# Diaries of The Upheaval - A Princess Zelda QA and Audio Response Bot

This project features a QA system in which **Princess Zelda** from *The Legend of Zelda* series responds to queries related to "Tears of the Kingdom." The database is extracted from Youtube Creator Zeltik's videos as following:

- [Zelda: Tears of the Kingdom - Story Explained part 1](https://www.youtube.com/watch?v=JuhBs44odO0) Duration: 1:21:23
- [Zelda: Tears of the Kingdom - Story Explained part 2](https://www.youtube.com/watch?v=qP1Fw2EpwqE) Duration: 1:02:31
- [Zelda: Tears of the Kingdom - Story Explained part 3](https://www.youtube.com/watch?v=JuhBs44odO0) Duration: 1:06:26
- [7 Secrets & Lore Details in Tears of the Kingdom](https://www.youtube.com/watch?v=w31M0LoVUO8) Duration: 13:16
- [Ganondorf’s Seal Explained - Zelda: Tears of the Kingdom Lore](https://www.youtube.com/watch?v=vad1wAe5mB4) Duration: 7:29
- [Tears of the Kingdom: A Disappointing Masterpiece](https://www.youtube.com/watch?v=Q1mRVn0WCrU)  Duration: 2:11:28
- [Ganondorf in Tears of the Kingdom: Lore, History & Speculation](https://www.youtube.com/watch?v=UhkwrgasKlU) Duration: 24:04

The bot is designed to stay in character, using a regal tone and retrieving relevant information from stored YouTube transcripts. With multilingual support, this project also includes voice synthesis to provide audio responses in Zelda’s voice.

## Features
- **Multilingual Support**: Responses available in multiple languages (English, Spanish, French, German, Portuguese, Italian).
- **Audio Transcription**: Whisper model used for audio transcription of user input.
- **Language Detection**: Language detection via `langdetect` to tailor responses appropriately.
- **Vector Database with ChromaDB**: Embedding-based search for accurate, contextually relevant responses.
- **Natural Language Processing with OpenAI API**: Uses GPT-3.5 to generate Princess Zelda responses in a character-consistent style.
- **Speech Synthesis with ElevenLabs**: Generates audio responses using ElevenLabs API for a more immersive experience.
- **Gradio Interface**: User-friendly interface for both text and audio inputs, with an enhanced Hyrule-themed design.

## Getting Started
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install Dependencies**: Install required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3. **API Keys**:
   - Store your OpenAI and Eleven Labs API keys in a `.env` file:
     ```env
     OPENAI_API_KEY=your_openai_key_here
     ELEVEN_LABS_API_KEY=your_elevenlabs_key_here
     ```

4. **Run the Application**: Launch the application using Gradio:
    ```bash
    python app.py
    ```

## Project Structure
- **Language-Based Prompts**: Custom prompts for each language to ensure accurate, in-character responses.
- **Video Transcript Processing**: Retrieves YouTube video transcripts to create a comprehensive, searchable database of text embeddings.
- **Text Chunking and Embedding**: Splits transcripts into manageable chunks and stores embeddings in a ChromaDB collection.
- **Multimodal QA System**: Integrates Whisper transcription and ElevenLabs TTS to provide both text and audio responses.
- **Evaluation Module**: Includes Giskard evaluation for response accuracy.

## Code Structure

### Key Functions and Components
- **Language Detection**:
    ```python
    def detect_language(text):
        try:
            return detect(text)
        except:
            return "en"  # Defaults to English if detection fails
    ```

- **Transcript Collection and Preprocessing**:
    ```python
    def get_transcript(video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB'])
            transcript_text = " ".join([item['text'] for item in transcript])
            return re.sub(r'\s+', ' ', transcript_text).strip()
        except Exception as e:
            print(f"Error retrieving transcript for video {video_id}: {e}")
            return None
    ```

- **Embedding Storage in ChromaDB**:
    ```python
    def store_transcript_embeddings(video_id, transcript_text):
        text_chunks = split_text(transcript_text)
        for i, chunk in enumerate(text_chunks):
            embedding = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")['data'][0]['embedding']
            chunk_id = f"{video_id}_chunk_{i}"
            collection.add(ids=[chunk_id], embeddings=[embedding], metadatas=[{'video_id': video_id, 'chunk_index': i, 'text': chunk}])
    ```

- **QA Model and Retrieval System**:
    ```python
    def generate_multi_query_response(user_query):
        detected_language = detect_language(user_query)
        prompt = multi_query_processing(user_query)
        zelda_formality = "Speak with reverence of the past, for the history of Hyrule is sacred..."
        enhanced_prompt = f"{get_prompt(detected_language)}\n\nQuestion: {user_query}\n\nDatabase Context:\n{prompt}\n\n{zelda_formality}"
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": enhanced_prompt}], max_tokens=150, temperature=0.7, stream=True)
        response_text = "".join(chunk['choices'][0].get('delta', {}).get('content', "") for chunk in response)
        return response_text
    ```

- **Gradio Interface Setup**:
    ```python
    gr_interface = gr.Interface(
        fn=handle_input,
        inputs=[gr.Textbox(label="What brings you here, beloved Hyrulean?"), gr.Audio(source="microphone", type="filepath", label="Or record your question")],
        outputs=[gr.Textbox(label="Response Text"), gr.Audio(label="Response Audio")],
        title="Diaries of The Upheaval",
        description="Ask me anything about the events of Tears of the Kingdom, and hear a response in Princess Zelda's voice."
    )
    ```

## Usage
1. **Ask Princess Zelda a Question**:
   - Input text or record audio to ask a question about "Tears of the Kingdom."
2. **Receive a Response**:
   - A text and audio response will be generated based on Zelda’s perspective, including contextually relevant information.
3. **Evaluation**:
   - Check responses using the built-in Giskard evaluation system to validate answers against expected responses.

## Contributing
1. **Fork the Project**.
2. **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`).
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`).
4. **Push to the Branch** (`git push origin feature/AmazingFeature`).
5. **Open a Pull Request**.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
Special thanks to:
- Zeltik for his great content on Legend of Zelda (visit https://www.youtube.com/@Zeltik)
- OpenAI for API access.
- ElevenLabs for voice synthesis.
- Gradio for a user-friendly interface.
  
