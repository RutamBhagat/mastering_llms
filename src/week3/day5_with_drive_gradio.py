# %% [markdown]
# # Meeting Minutes Generation from Google Drive Audio
#
# This script provides a streamlined way to generate meeting minutes from an audio file stored in Google Drive. The steps are:
#
# 1. Authenticate with Google Drive.
# 2. List audio files from a specified folder.
# 3. Allow users to select a file via a Gradio interface.
# 4. Download the selected file.
# 5. Process the file using OpenAI Whisper for transcription.

# %% [markdown]
# ## Step 1: Install Required Libraries
#
# Make sure you have all the necessary libraries installed.

# %%
# !pip install -q google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client gradio openai torch transformers

# %% [markdown]
# ## Step 2: Google Drive Authentication
#
# This script uses OAuth2 for secure access to the user's Google Drive. Download the credentials JSON from your Google Cloud Console and save it as `client_secrets.json` in this script's directory.

# %%
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define the scopes and paths
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_PATH = "token.json"
CREDENTIALS_PATH = "client_secrets.json"


def authenticate_drive():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    service = build("drive", "v3", credentials=creds)
    return service


# Authenticate and create a Google Drive service instance
drive_service = authenticate_drive()

# %% [markdown]
# ## Step 3: List Audio Files in Google Drive
#
# This function queries the Google Drive API for audio files (e.g., `mp3`, `wav` formats) in a specified folder.


# %%
def list_audio_files(service, folder_id=None):
    query = "mimeType='audio/mpeg' or mimeType='audio/wav'"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )
    files = results.get("files", [])
    return {file["name"]: file["id"] for file in files}


# Get audio files (use folder_id if you want to specify a specific folder)
audio_files = list_audio_files(drive_service)
audio_files

# %% [markdown]
# ## Step 4: Download and Process Audio File
#
# The `download_file` function retrieves the selected audio file. After downloading, `process_audio` handles transcription.

# %%
import requests


def download_file(file_id):
    request_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {
        "Authorization": f"Bearer {drive_service._http.request.credentials.token}"
    }
    response = requests.get(request_url, headers=headers)
    file_path = f"{file_id}.mp3"
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def process_audio(file_id):
    file_path = download_file(file_id)
    # Example: Implement transcription here using OpenAI or Whisper
    transcription_result = f"Transcription for {file_path}"
    return transcription_result


# Test with one of the file IDs from `audio_files`
example_file_id = list(audio_files.values())[0] if audio_files else None
if example_file_id:
    process_audio(example_file_id)

# %% [markdown]
# ## Step 5: Gradio UI for Audio Selection and Transcription
#
# Gradio serves as the user interface, allowing users to select an audio file from their Google Drive and receive the transcription in Markdown format.

# %%
import gradio as gr


def interface():
    files_dict = list_audio_files(drive_service)
    dropdown = gr.Dropdown(list(files_dict.keys()), label="Select Audio File")

    def on_file_select(file_name):
        file_id = files_dict[file_name]
        result = process_audio(file_id)
        return result

    gr.Interface(
        fn=on_file_select,
        inputs=dropdown,
        outputs="markdown",
        title="Meeting Minutes from Google Drive Audio",
    ).launch()


# Run the interface
interface()
