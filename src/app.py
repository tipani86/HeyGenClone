import os
import io
import av
import base64
import asyncio
import requests
import pandas as pd
from utils import *
import streamlit as st
from stqdm import stqdm
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI

UTC_TIMESTAMP = int(datetime.utcnow().timestamp())

client = AsyncOpenAI()


@st.cache_data(show_spinner=False)
def extract_audio(video_file):
    # Use PyAV to open the input video file, extract the audio stream, output to memory buffer and return it
    video_container = av.open(video_file)
    audio_in = video_container.streams.get(audio=0)[0]

    audio_buffer = io.BytesIO()
    with av.open(audio_buffer, "w", "mp3") as audio_container:
        audio_out = audio_container.add_stream("mp3")

        # Get the length of all frames in the video for the progress bar
        total_frames = audio_in.frames

        for frame in stqdm(video_container.decode(audio_in), total=total_frames, desc="Extracting audio...", mininterval=1):
            frame.pts = None
            for packet in audio_out.encode(frame):
                audio_container.mux(packet)
        
        for packet in audio_out.encode(None):
            audio_container.mux(packet)

    audio_buffer.seek(0)
    return audio_buffer


@st.cache_data(show_spinner=False)
def build_voices_dataframe(
    voices: dict
) -> pd.DataFrame:
    voices_copy = voices.copy()
    # Move any sub key-value pairs from under "labels" key to the root level, also delete the key "fine_tuning"
    for voice in voices_copy.values():
        voice.update(voice.pop("labels"))
        voice.pop("fine_tuning")
    return pd.DataFrame(voices_copy.values())[["name", "category", "age", "gender", "accent", "description", "use case"]].set_index("name")


@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
async def recognize_speech(
    audio: io.BytesIO
) -> str:
    return "TEST"
    if "RECOGNIZED_TEXT" in st.session_state:
        return st.session_state["RECOGNIZED_TEXT"]
    
    file_size = len(audio.getvalue())
    if file_size > 25 * 1024 * 1024:
        st.error(f"Audio file size exceeds 25MB limit. Please upload a smaller file.")
        st.stop()

    # Call the OpenAI API to recognize speech
    transcript = await client.audio.transcriptions.create(
        model="whisper-1",  # Only this one available for now
        file=("audio.mp3", audio)
    )
    try:
        transcript = transcript.text
    except:
        transcript = transcript["text"] if "text" in transcript else transcript
    return transcript


@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
async def translate_text(
    src_text: str,
    dst_lang: str,
) -> str:
    if "TRANSLATED_TEXT" in st.session_state:
        return st.session_state["TRANSLATED_TEXT"]
    
    # Call the OpenAI API to translate text
    messages = [
        {"role": "system", "content": f"You are a highly skilled professional translator that understands every nuance of different languages and can create translations from user-provided source material that are indistinguishable from native speakers. The source material might come from an audio transcription (either edited or not) so it may require some cleaning or best effort interpretation. Skip filler words like 'um', 'erm' to create a professional, flowing translation instead. Source language: [ Auto-Detect ] Target language: [ {dst_lang} ]"},
        {"role": "user", "content": src_text}
    ]
    response = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        max_tokens=4000,
    )
    return response.choices[0].message.content.strip()


async def get_voices():
    # Get a list of currently available voices
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
    }
    voice_list = []
    async for response in call_api("GET", url, headers=headers):
        voice_list.extend(response["voices"])
    voices = {voice["name"]: voice for voice in voice_list}
    return voices


@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
async def create_voice_clone(
    name: str,
    description: str,
    audio: io.BytesIO,
) -> str:
    # Create a voice clone with an audio file (in memory)
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
    }
    # Convert the input audio into an UploadFile object
    data = {
        "name": name,
        "description": description,
    }
    files = [
        ("files", ("audio.mp3", audio, "audio/mpeg"))
    ]
    response = requests.request("POST", url, headers=headers, data=data, files=files)
    if response.status_code != 200:
        raise RuntimeError(f"API call failed with status {response.status_code}: {response.text}")
    return response.json()["voice_id"]


@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
async def generate_voice(
    voice_id: str,
    text: str,
) -> bytes:
    if "GENERATED_AUDIO" in st.session_state:
        return st.session_state["GENERATED_AUDIO"]
    
    # Generate a spoken speech audio file (in memory) from a text string
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
    }
    data = {
        "model_id": "eleven_multilingual_v2",
        "text": text,
    }
    async for response in call_api("POST", url, headers=headers, data=data):
        audio = response
    return audio


def create_download_link(data, filename):
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode()
    else:
        b64 = base64.b64encode(data.encode()).decode()
    ext = Path(filename).suffix[1:]
    href = f'<a href="data:file/{ext};base64,{b64}" download="{filename}">Download as {ext} file</a>'
    return href


def reset_session_state():
    if "RECOGNIZED_TEXT" in st.session_state:
        del st.session_state["RECOGNIZED_TEXT"]
    if "TRANSLATED_TEXT" in st.session_state:
        del st.session_state["TRANSLATED_TEXT"]
    if "GENERATED_AUDIO" in st.session_state:
        del st.session_state["GENERATED_AUDIO"]


############
### MAIN ###
############


async def main():
    st.set_page_config(
        page_title="HeyGen Clone",
        page_icon="🎞️",
        menu_items=None,
    )

    env_keys = ["ELEVEN_API_KEY", "OPENAI_API_KEY"]
    errors = []
    for key in env_keys:
        if key not in os.environ:
            errors.append(key)
    if errors:
        st.error(f"Please set the following environment variables: {', '.join(errors)}")
        st.stop()

    st.title("HeyGen Clone")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        reset_session_state()

    step_1 = st.container(border=True)
    with step_1:
        st.write("**Step 1: Upload a (mp4) video**")
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_file is None:
        # No file has been uploaded, clear any downstream session_state items
        reset_session_state()
        st.stop()

    if "UPLOADED_FILE" not in st.session_state:
        st.session_state["UPLOADED_FILE"] = uploaded_file.name
    
    if "UPLOADED_FILE" in st.session_state and st.session_state["UPLOADED_FILE"] != uploaded_file.name:
        # A different file has been uploaded, clear any downstream session_state items
        reset_session_state()

    with step_1:
        with st.expander("Preview Video"):
            st.video(uploaded_file)

    step_2 = st.container(border=True)
    with step_2:
        st.write("**Step 2: Extract audio from video and transcribe it**")
        audio = extract_audio(uploaded_file)
        with st.expander("Preview Audio"):
            st.audio(audio, format="audio/mp3")
        send_to_whisper = st.button("Send to Whisper for transcription")
    
    if "RECOGNIZED_TEXT" not in st.session_state and not send_to_whisper:
        st.stop()

    tasks = [get_voices(), recognize_speech(audio)]
    with st.spinner("Loading..."):
        voices, recognized_text = await asyncio.gather(*tasks)

    if "RECOGNIZED_TEXT" not in st.session_state:
        st.session_state["RECOGNIZED_TEXT"] = recognized_text

    step_3 = st.expander("**Step 3 (Optional): Create a Voice Clone**", expanded=False)
    with step_3:
        st.dataframe(build_voices_dataframe(voices), use_container_width=True, height=200)
        voice_name = st.selectbox("Select a voice to hear a preview", list(voices.keys()), index=None)
        if voice_name is not None:
            voice = voices[voice_name]
            st.audio(voice["preview_url"], format="audio/mp3")
            st.json(voice, expanded=False)
        with st.form("Create Voice Clone"):
            st.write("Create a voice clone with the extracted audio from Step 2")
            name_col, desc_col = st.columns([1, 2])
            with name_col:
                name = st.text_input("Name")
            with desc_col:
                description = st.text_input("Description")
            submit_voice_clone = st.form_submit_button("Submit")
        if submit_voice_clone:
            with st.spinner("Creating voice clone..."):
                st.session_state["VOICE_CLONE_ID"] = await create_voice_clone(name, description, audio)
            
    step_4 = st.container(border=True)
    with step_4:
        st.write("**Step 4: Translate text**")
        with st.form("Edit for translation"):
            src_text = st.text_area("Recognized text", st.session_state["RECOGNIZED_TEXT"])
            dst_lang = st.text_input("Target Language")
            submit_translation = st.form_submit_button("Submit")

    if "TRANSLATED_TEXT" not in st.session_state and not submit_translation:
        st.stop()

    if src_text != st.session_state["RECOGNIZED_TEXT"]:
        # User has edited the recognized text, update it and clear any downstream session_state items
        st.session_state["RECOGNIZED_TEXT"] = src_text
        if "TRANSLATED_TEXT" in st.session_state:
            del st.session_state["TRANSLATED_TEXT"]
        if "GENERATED_AUDIO" in st.session_state:
            del st.session_state["GENERATED_AUDIO"]

    with st.spinner("Translating text..."):
        translated_text = await translate_text(src_text, dst_lang)

    if "TRANSLATED_TEXT" not in st.session_state:
        st.session_state["TRANSLATED_TEXT"] = translated_text

    step_5 = st.container(border=True)
    with step_5:
        st.write("**Step 5: Generate translated audio track with**")
        with st.form("Edit for voice generation"):
            dst_text = st.text_area("Translated text", st.session_state["TRANSLATED_TEXT"])
            voice_name = st.selectbox("Select a voice", list(voices.keys()), index=None)
            submit_voice_generation = st.form_submit_button("Submit")
        
    if "GENERATED_AUDIO" not in st.session_state and not submit_voice_generation:
        st.stop()
    
    if dst_text != st.session_state["TRANSLATED_TEXT"]:
        # User has edited the translated text, update it and clear any downstream session_state items
        st.session_state["TRANSLATED_TEXT"] = dst_text
        if "GENERATED_AUDIO" in st.session_state:
            del st.session_state["GENERATED_AUDIO"]

    with st.spinner("Generating audio..."):
        audio = await generate_voice(voices[voice_name]["voice_id"], translated_text)
    
    if "GENERATED_AUDIO" not in st.session_state:
        st.session_state["GENERATED_AUDIO"] = {
            "audio": audio,
            "name": voice_name,
            "timestamp": UTC_TIMESTAMP,
        }

    step_6 = st.container(border=True)
    with step_6:
        st.write("**Step 6: Preview translated audio track**")
        with st.expander("Preview Audio"):
            st.audio(st.session_state["GENERATED_AUDIO"]["audio"], format="audio/mp3")
        st.markdown(
            create_download_link(
                st.session_state["GENERATED_AUDIO"]["audio"],
                f'generated_voice_{st.session_state["GENERATED_AUDIO"]["name"]}_{st.session_state["GENERATED_AUDIO"]["timestamp"]}.mp3',
            ),
            unsafe_allow_html=True
        )
        submit_lip_syncing = st.button("Submit to Lip Syncing")
    
    if not submit_lip_syncing:
        st.stop()

if __name__ == "__main__":
    asyncio.run(main())