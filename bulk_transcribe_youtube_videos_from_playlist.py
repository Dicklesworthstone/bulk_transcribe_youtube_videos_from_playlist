import os
import asyncio
import re
import sys
import psutil
import glob
import openai
import json
import datetime
import traceback
from pytubefix import YouTube, Playlist
import pandas as pd
from faster_whisper import WhisperModel
from numba import cuda
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Constants for pricing
WHISPER_COST_PER_MINUTE = 0.006

convert_single_video = 1
use_spacy_for_sentence_splitting = 1
use_openai_api_for_transcription = 0
openai_api_key = 'REPLACE_WITH_YOUR_API_KEY'
max_simultaneous_youtube_downloads = 4
disable_cuda_override = 0
single_video_url = 'https://www.youtube.com/watch?v=sWAaJF9Wk0w'
playlist_url = 'https://www.youtube.com/playlist?list=PLjpPMe3LP1XKgqqzqz4j6M8-_M_soYxiV'

if convert_single_video:
    print(f"Processing a single video: {single_video_url}")
else:
    print(f"Processing a playlist: {playlist_url}")

def add_to_system_path(new_path):
    if new_path not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = new_path + os.pathsep + os.environ["PATH"]
    if sys.platform == "win32" and ' ' in new_path and not new_path.startswith('"') and not new_path.endswith('"'):
        os.environ["PATH"] = f'"{new_path}"' + os.pathsep + os.environ["PATH"].replace(new_path, "")

def get_cuda_toolkit_path():
    home_dir = os.path.expanduser('~')
    if sys.platform in ["win32", "linux", "linux2", "darwin"]:
        anaconda_base_path = os.path.join(home_dir, "anaconda3", "pkgs")
    cuda_glob_pattern = os.path.join(anaconda_base_path, "cudatoolkit-*", "Library", "bin")
    cuda_paths = glob.glob(cuda_glob_pattern)
    if cuda_paths:
        return cuda_paths[0]
    return None

cuda_toolkit_path = get_cuda_toolkit_path()
print("CUDA Toolkit Path:", cuda_toolkit_path)
if cuda_toolkit_path:
    add_to_system_path(cuda_toolkit_path)

max_workers_transcribe = psutil.cpu_count(logical=False)  # Number of physical cores

os.makedirs('downloaded_audio', exist_ok=True)
os.makedirs('generated_transcript_combined_texts', exist_ok=True)
os.makedirs('generated_transcript_metadata_tables', exist_ok=True)

if use_spacy_for_sentence_splitting:
    import spacy
    import spacy.cli
    def download_spacy_model(model_name="en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)
    nlp = download_spacy_model()  
    def sophisticated_sentence_splitter(text):
        text = remove_pagination_breaks(text)
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences        
else:    
    def sophisticated_sentence_splitter(text):
        text = remove_pagination_breaks(text)
        pattern = r'\.(?!\s*(com|net|org|io)\s)(?![0-9])'
        pattern += r'|[.!?]\s+'
        pattern += r'|\.\.\.(?=\s)'
        sentences = re.split(pattern, text)
        refined_sentences = []
        temp_sentence = ""
        for sentence in sentences:
            if sentence is not None:
                temp_sentence += sentence
                if temp_sentence.count('"') % 2 == 0:
                    refined_sentences.append(temp_sentence.strip())
                    temp_sentence = ""
        if temp_sentence:
            refined_sentences.append(temp_sentence.strip())
        return [s.strip() for s in refined_sentences if s.strip()]

def clean_filename(title):
    title = re.sub(r'[^\w\s-]', '', title)
    return re.sub(r'[-\s]+', '_', title).strip().lower()

async def download_audio(video):
    filename = clean_filename(video.title)
    base_filename = filename
    counter = 1
    audio_dir = 'downloaded_audio'
    audio_file_path = os.path.join(audio_dir, f"{filename}.mp4")
    while os.path.exists(audio_file_path):
        filename = f"{base_filename}_{counter}"
        audio_file_path = os.path.join(audio_dir, f"{filename}.mp4")
        counter += 1
    if not os.path.exists(audio_file_path):
        stream = video.streams.filter(only_audio=True).first()
        if stream is None:
            raise ValueError(f"No audio stream found for video: {video.title}")
        try:
            os.makedirs(audio_dir, exist_ok=True)
            audio_file_path = stream.download(output_path=audio_dir, filename=f"{filename}.mp4")
        except Exception as e:
            print(f"Error downloading video {video.title}: {e}")
            return None, None
    return audio_file_path, filename

def estimate_whisper_transcription_cost(audio_duration_seconds):
    audio_duration_minutes = audio_duration_seconds / 60
    estimated_cost = audio_duration_minutes * WHISPER_COST_PER_MINUTE
    print(f"\n=== Estimated Whisper Transcription Cost ===")
    print(f"Audio Duration: {audio_duration_minutes:.2f} minutes")
    print(f"Estimated Cost: ${estimated_cost:.4f}\n")
    return estimated_cost

async def get_audio_duration(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000  # pydub returns duration in milliseconds

def remove_unwanted_segments_from_json(json_file_path, unwanted_text="Subtitles by the Amara.org community"):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    original_segment_count = len(data)
    filtered_data = [segment for segment in data if unwanted_text not in segment['text']]
    filtered_segment_count = len(filtered_data)
    if original_segment_count != filtered_segment_count:
        with open(json_file_path, 'w') as json_file:
            json.dump(filtered_data, json_file, indent=4)
        print(f"Removed {original_segment_count - filtered_segment_count} unwanted segments from {json_file_path}.")
    else:
        print(f"No unwanted segments found in {json_file_path}.")
        
async def compute_transcript_with_whisper_from_audio_func(audio_file_path, audio_file_name, audio_file_size_mb):
    cuda_toolkit_path = get_cuda_toolkit_path()
    if cuda_toolkit_path:
        add_to_system_path(cuda_toolkit_path)
    combined_transcript_text = ""
    combined_transcript_text_list_of_metadata_dicts = []
    list_of_transcript_sentences = []
    if use_openai_api_for_transcription:
        print(f"Using OpenAI API for transcription of {audio_file_name}...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        audio_duration_seconds = await get_audio_duration(audio_file_path)
        estimate_whisper_transcription_cost(audio_duration_seconds)
        with open(audio_file_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        # Access the response content
        response_data = json.loads(response.model_dump_json())
        segments = response_data.get('segments', [])
        combined_transcript_text = response_data.get('text', "")
        combined_transcript_text_list_of_metadata_dicts = [
            {
                "start": segment.get('start', 0),
                "end": segment.get('end', 0),
                "text": segment.get('text', ""),
                "avg_logprob": segment.get('avg_logprob', 0)
            }
            for segment in segments
        ]
    else:
        print(f"Using local Whisper model for transcription of {audio_file_name}...")
        if cuda.is_available() and not disable_cuda_override:
            print("CUDA is available. Using GPU for transcription.")
            device = "cuda"
            compute_type = "float16"
        else:
            print("CUDA not available. Using CPU for transcription.")
            device = "cpu"
            compute_type = "auto"
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        request_time = datetime.datetime.now(datetime.UTC)
        print(f"Computing transcript for {audio_file_name} which has a {audio_file_size_mb:.2f}MB file size...")
        audio_duration_seconds = await get_audio_duration(audio_file_path)
        with tqdm(total=audio_duration_seconds, desc=f"Transcribing {audio_file_name}", unit="s") as pbar:
            segments, info = await asyncio.to_thread(model.transcribe, audio_file_path, beam_size=10, vad_filter=True)
            for segment in segments:
                pbar.update(segment.end - segment.start)
                print(f"Processing segment: [Start: {segment.start:.2f}s, End: {segment.end:.2f}s] for file {audio_file_name} with text: {segment.text}")
                combined_transcript_text += segment.text + " "
                sentences = sophisticated_sentence_splitter(segment.text)
                list_of_transcript_sentences.extend(sentences)
                metadata = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text,
                    "avg_logprob": round(segment.avg_logprob, 2)
                }
                combined_transcript_text_list_of_metadata_dicts.append(metadata)
    if not combined_transcript_text_list_of_metadata_dicts:
        print(f"No segments were returned for file {audio_file_name}.")
        return [], {}, "", [], datetime.datetime.now(datetime.UTC), datetime.datetime.now(datetime.UTC), 0, ""
    with open(f'generated_transcript_combined_texts/{audio_file_name}.txt', 'w') as file:
        file.write(combined_transcript_text)
    df = pd.DataFrame(combined_transcript_text_list_of_metadata_dicts)
    df.to_csv(f'generated_transcript_metadata_tables/{audio_file_name}.csv', index=False)
    json_file_path = f'generated_transcript_metadata_tables/{audio_file_name}.json'
    df.to_json(json_file_path, orient='records', indent=4)
    remove_unwanted_segments_from_json(json_file_path)   
    return combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, list_of_transcript_sentences

async def process_video_or_playlist(url, max_simultaneous_downloads, max_workers_transcribe):
    if convert_single_video:
        yt = YouTube(url)
        videos = [yt]
    else:
        playlist = Playlist(url)
        videos = playlist.videos
    download_semaphore = asyncio.Semaphore(max_simultaneous_downloads)
    async def download_and_transcribe(video):
        try:
            async with download_semaphore:
                audio_path, audio_filename = await download_audio(video)
                if audio_path and audio_filename:
                    audio_file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    await compute_transcript_with_whisper_from_audio_func(audio_path, audio_filename, audio_file_size_mb)
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing video {video.title}: {e}")
    tasks = [download_and_transcribe(video) for video in videos]
    await asyncio.gather(*tasks)

def normalize_logprobs(avg_logprob, min_logprob, max_logprob):
    range_logprob = max_logprob - min_logprob
    return (avg_logprob - min_logprob) / range_logprob if range_logprob != 0 else 0.5

def remove_pagination_breaks(text: str) -> str:
    text = re.sub(r'-(\n)(?=[a-z])', '', text)
    text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text)
    return text

def merge_transcript_segments_into_combined_text(segments):
    if not segments:
        return "", [], []
    min_logprob = min(segment['avg_logprob'] for segment in segments)
    max_logprob = max(segment['avg_logprob'] for segment in segments)
    combined_text = ""
    sentence_buffer = ""
    list_of_metadata_dicts = []
    list_of_sentences = []
    char_count = 0
    time_start = None
    time_end = None
    total_logprob = 0.0
    segment_count = 0
    for segment in segments:
        if time_start is None:
            time_start = segment['start']
        time_end = segment['end']
        total_logprob += segment['avg_logprob']
        segment_count += 1
        sentence_buffer += segment['text'] + " "
        sentences = sophisticated_sentence_splitter(sentence_buffer)
        for sentence in sentences:
            combined_text += sentence.strip() + " "
            list_of_sentences.append(sentence.strip())
            char_count += len(sentence.strip()) + 1
            avg_logprob = total_logprob / segment_count
            model_confidence_score = normalize_logprobs(avg_logprob, min_logprob, max_logprob)
            metadata = {
                'start_char_count': char_count - len(sentence.strip()) - 1,
                'end_char_count': char_count - 2,
                'time_start': time_start,
                'time_end': time_end,
                'model_confidence_score': model_confidence_score
            }
            list_of_metadata_dicts.append(metadata)
        if sentences:
            sentence_buffer = sentences.pop() if len(sentences) % 2 != 0 else ""
    return combined_text, list_of_metadata_dicts, list_of_sentences

if __name__ == '__main__':
    url_to_process = single_video_url if convert_single_video else playlist_url
    asyncio.run(process_video_or_playlist(url_to_process, max_simultaneous_youtube_downloads, max_workers_transcribe))
