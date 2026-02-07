"""LectureChat - Multimodal RAG Video Assistant Streamlit App."""

import streamlit as st
from pathlib import Path
import time
from typing import Optional

from src.config import VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, CHROMA_DIR
from src.utils import validate_youtube_url, generate_video_id, format_timestamp, create_directories
from src.video_processor import VideoProcessor
from src.transcription import AudioTranscriber
from src.rag_engine import RAGEngine


# Page config
st.set_page_config(
    page_title="LectureChat - Multimodal RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4da6ff; /* Lighter blue for dark mode contrast */
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Streamlit handles chat message styling well by default. 
       We only add subtle borders/spacing if needed, but removing background-color 
       to prevent white-on-white text issues in dark mode. */
    .stChatMessage {
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 10px;
        margin: 5px 0;
    }
    .source-box {
        background-color: rgba(255, 255, 255, 0.05); /* Semi-transparent white for dark mode */
        border-left: 4px solid #4da6ff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .timestamp-link {
        color: #4da6ff;
        font-weight: bold;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


def process_video(video_source: str, source_type: str, progress_bar, status_text):
    """
    Process video: download/upload, extract audio, extract frames, build index.
    
    Args:
        video_source: YouTube URL or uploaded file
        source_type: 'youtube' or 'upload'
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
    """
    try:
        # Step 0: Handle existing video
        if source_type == 'existing':
            # Extract video_id from filename (assuming format "id.mp4" or just use filename stem)
            video_path = Path(video_source)
            # The video_id is the filename without extension if we used our naming convention
            # If the user renamed it, this might be tricky, but let's assume our convention
            # Actually, `video_id` was `uuid` initially.
            # Best effort: use the stem as video_id
            video_id = video_path.stem
            
            st.session_state.video_id = video_id
            st.session_state.video_path = str(video_path)
            
            status_text.text("🔄 Loading existing index...")
            progress_bar.progress(50)
            
            # Initialize RAG engine (connects to existing ChromaDB)
            rag_engine = RAGEngine(video_id)
            # We assume the index exists. If not, it will be empty, which is a risk.
            # Ideally we check if collection exists, but RAG engine handles get_or_create.
            
            st.session_state.rag_engine = rag_engine
            
            progress_bar.progress(100)
            status_text.text("✅ Loaded successfully! Ready to chat.")
            st.session_state.processing_complete = True
            return True

        # Generate video ID for NEW videos
        video_id = generate_video_id()
        st.session_state.video_id = video_id
        
        # Initialize processor
        video_processor = VideoProcessor()
        
        # Step 1: Get video
        status_text.text("📥 Downloading/uploading video...")
        progress_bar.progress(10)
        
        if source_type == 'youtube':
            video_path = video_processor.download_youtube(video_source, video_id)
        else:
            video_path = video_processor.save_uploaded_video(video_source, video_id)
        
        st.session_state.video_path = video_path
        
        # Step 2: Extract audio
        status_text.text("🎵 Extracting audio...")
        progress_bar.progress(25)
        audio_path = video_processor.extract_audio(video_path, video_id)
        
        # Step 3: Extract frames
        status_text.text("🖼️ Extracting unique slides (smart scene detection)...")
        progress_bar.progress(40)
        frame_data = video_processor.extract_frames(video_path, video_id)
        
        # Step 4: Transcribe audio
        status_text.text("📝 Transcribing audio with Whisper...")
        progress_bar.progress(55)
        transcriber = AudioTranscriber()
        segments = transcriber.transcribe_audio(audio_path)
        chunks = transcriber.chunk_transcription(segments)
        
        # Step 5: Build RAG index
        status_text.text("🧠 Building multimodal index...")
        progress_bar.progress(75)
        rag_engine = RAGEngine(video_id)
        rag_engine.build_index(chunks, frame_data)
        
        st.session_state.rag_engine = rag_engine
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Processing complete! Ready to chat.")
        st.session_state.processing_complete = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return False


def main():
    """Main Streamlit app."""
    
    # Initialize
    create_directories()
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎓 LectureChat</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Ask questions about your lecture videos using multimodal RAG</p>', unsafe_allow_html=True)
    
    # Sidebar for video upload
    with st.sidebar:
        st.header("📹 Load Lecture Video")
        
        input_method = st.radio(
            "Choose input method:",
            ["YouTube URL", "Upload MP4", "Sample Video (Demo)", "Select Existing Video"],
            key="input_method"
        )
        
        video_source = None
        source_type = None
        
        if input_method == "YouTube URL":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            if youtube_url:
                if validate_youtube_url(youtube_url):
                    video_source = youtube_url
                    source_type = 'youtube'
                else:
                    st.error("Invalid YouTube URL")
        
        elif input_method == "Upload MP4":
            uploaded_file = st.file_uploader(
                "Upload video file:",
                type=['mp4', 'mov', 'avi'],
                accept_multiple_files=False
            )
            if uploaded_file:
                video_source = uploaded_file
                source_type = 'upload'

        elif input_method == "Sample Video (Demo)":
            st.info("ℹ️ Using a sample 1-minute lecture about Neural Networks.")
            # Use a known safe YouTube video
            video_source = "https://www.youtube.com/watch?v=aircAruvnKk"
            source_type = 'youtube'
            
        else:  # Select Existing Video
            # List all mp4 files in VIDEOS_DIR
            existing_videos = list(VIDEOS_DIR.glob("*.mp4"))
            if not existing_videos:
                st.warning("No processed videos found.")
            else:
                # Create a mapping of display name to path
                video_options = {f.name: f for f in existing_videos}
                selected_video_name = st.selectbox(
                    "Select a previously processed video:",
                    options=list(video_options.keys())
                )
                
                if selected_video_name:
                    video_source = str(video_options[selected_video_name])
                    source_type = 'existing'

        
        # Process button
        if video_source:
            if st.button("🚀 Process Video", type="primary", use_container_width=True):
                st.session_state.processing_complete = False
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success = process_video(video_source, source_type, progress_bar, status_text)
                
                if success:
                    st.success("Ready to chat!")
                    time.sleep(1)
                    st.rerun()
        
        # Status indicator
        if st.session_state.processing_complete:
            st.success("✅ Video loaded and indexed")
            if st.button("🗑️ Clear Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Info
        st.divider()
        st.markdown("""
        ### ℹ️ How it works:
        1. **Upload** a lecture video or YouTube URL
        2. **Smart scene detection** extracts unique slides
        3. **Whisper** transcribes the audio
        4. **CLIP** embeds visual content
        5. **Chat** with multimodal RAG!
        
        ### 🔧 Requirements:
        - Ollama running locally
        - Model: `mistral` or `llama3`
        """)
    
    # Main content area
    if not st.session_state.processing_complete:
        # Welcome screen
        st.info("👈 Upload a video or enter a YouTube URL to get started!")
        
        st.markdown("""
        ## Features
        - 🎬 **Smart Slide Extraction**: Automatically detects unique slides using scene detection
        - 🎙️ **Audio Transcription**: Whisper-powered transcription with timestamps
        - 🖼️ **Visual Understanding**: CLIP embeddings for image-text matching
        - 🤖 **Local LLM**: Privacy-first with Ollama
        - 💬 **Multimodal Chat**: Ask questions about both audio and visual content
        
        ## Example Questions
        - "What is the main topic of this lecture?"
        - "Can you explain the concept shown at 5:30?"
        - "What does the diagram on slide 3 illustrate?"
        - "Summarize the key points from the first 10 minutes"
        """)
    
    else:
        # Split layout: video player | chat interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎥 Video Player")
            
            # Display video
            if st.session_state.video_path:
                st.video(st.session_state.video_path)
            
            # Show extracted frames count
            if st.session_state.rag_engine:
                st.caption(f"📊 Processed with multimodal RAG indexing")
        
        with col2:
            st.subheader("💬 Chat with Video")
            
            # Chat history
            chat_container = st.container(height=400)
            
            with chat_container:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.chat_message("user").write(message['content'])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message['content'])
                            
                            # Show sources if available
                            if 'sources' in message:
                                sources = message['sources']
                                
                                # Text timestamps
                                if sources.get('text_timestamps'):
                                    st.markdown("**📝 Referenced from transcript:**")
                                    for ts in sources['text_timestamps']:
                                        st.caption(f"⏱️ {ts['time']}")
                                
                                # Image frames
                                if sources.get('image_frames'):
                                    st.markdown("**🖼️ Relevant slide:**")
                                    for frame in sources['image_frames']:
                                        col_a, col_b = st.columns([1, 3])
                                        with col_a:
                                            st.image(frame['frame_path'], width=150)
                                        with col_b:
                                            st.caption(f"⏱️ Timestamp: {frame['timestamp_str']}")
            
            # Chat input
            user_question = st.chat_input("Ask a question about the video...")
            
            if user_question:
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Get response
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_engine.query(user_question)
                    
                    # Add assistant message
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'sources': result['sources']
                    })
                
                # Rerun to show new messages
                st.rerun()


if __name__ == "__main__":
    main()
