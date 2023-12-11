import replicate
import streamlit as st

st.set_page_config("Replicate Test", layout="wide")

st.title("Replicate Test")

video, audio = st.columns(2)

with video:
    video_file = st.file_uploader("Upload video", type=["mp4"])
with audio:
    audio_file = st.file_uploader("Upload audio", type=["wav"])

if not video_file or not audio_file:
    st.stop()

run_lip_sync = st.button("Run Lip Sync")

if not run_lip_sync:
    st.stop()

with st.spinner("Generating Lip Sync..."):
    output = replicate.run(
        "cjwbw/video-retalking:ecd06c5e9ceed2e3e061b44fb852240c5a24bb902db08061b55f7f85a4d0cbe2",
        input={
            "face": "https://www.dropbox.com/scl/fi/7zqy6bld579of1azx3ahv/Nordea_Test.mp4?rlkey=jhjpoydz8tq2j1wxn3jihniwe&dl=1",
            "input_audio": "https://www.dropbox.com/scl/fi/ni0jx0s1g6ayz63k3dk6n/generated_voice_Tianyi_1702012661.mp3?rlkey=ubfs1ef9vpls6ihfxom0x7l78&dl=1"
        }
    )

st.video(output["output"])
