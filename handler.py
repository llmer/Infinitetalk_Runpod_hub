import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import sys
import urllib.request
import urllib.parse
import binascii
import subprocess
import threading
import librosa
import shutil
import time
import re

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def truncate_base64_for_log(base64_str, max_length=50):
    """Truncate a base64 string for logging."""
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... ({len(base64_str)} chars total)"


server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())
RESULT_CACHE_DIR = "/tmp/runpod_job_cache"


def get_request_key(job):
    """Extract a stable request identifier for idempotent retries."""
    request_id = (
        job.get("id")
        or job.get("requestId")
        or job.get("request_id")
        or job.get("uid")
    )
    if not request_id:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(request_id))


def get_cache_paths(request_key):
    base_path = os.path.join(RESULT_CACHE_DIR, request_key)
    return {
        "meta": f"{base_path}.json",
        "video": f"{base_path}.mp4",
    }


def save_cached_video(request_key, source_video_path):
    """Persist a copy of the generated video for idempotent retries."""
    if not request_key:
        return
    try:
        os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
        paths = get_cache_paths(request_key)
        temp_video_path = f"{paths['video']}.tmp"
        shutil.copy2(source_video_path, temp_video_path)
        os.replace(temp_video_path, paths["video"])
        with open(paths["meta"], "w") as f:
            json.dump(
                {
                    "request_key": request_key,
                    "video_path": paths["video"],
                },
                f,
            )
        logger.info(f"Saved idempotency cache for request_key={request_key}")
    except Exception as e:
        logger.warning(f"Failed to save idempotency cache ({request_key}): {e}")


def load_cached_result(request_key, use_network_volume):
    """Return cached output for requeued jobs if available."""
    if not request_key:
        return None

    paths = get_cache_paths(request_key)
    if not os.path.exists(paths["meta"]) or not os.path.exists(paths["video"]):
        return None

    if use_network_volume:
        network_volume_dir = "/runpod-volume"
        if os.path.isdir(network_volume_dir):
            output_path = os.path.join(network_volume_dir, f"infinitetalk_{request_key}.mp4")
            shutil.copy2(paths["video"], output_path)
            logger.info(f"Returning cached network volume video for request_key={request_key}")
            return {"video_path": output_path}
        logger.warning("network_volume requested but /runpod-volume is unavailable; using base64")

    with open(paths["video"], "rb") as f:
        video_data = base64.b64encode(f.read()).decode("utf-8")
    logger.info(f"Returning cached base64 video for request_key={request_key}")
    return {"video": video_data}


def download_file_from_url(url, output_path):
    """Download a file from a URL."""
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30", url],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(f"Downloaded file: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode base64 data and save to a file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, "wb") as f:
            f.write(decoded_data)

        logger.info(f"Saved base64 input to '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")


def process_input(input_data, temp_dir, output_filename, input_type):
    """Process input data and return a file path."""
    if input_type == "path":
        logger.info(f"Processing path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"Processing URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Processing base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def queue_prompt(prompt, input_type="image", person_count="single"):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    logger.info(f"Workflow node count: {len(prompt)}")
    if input_type == "image":
        logger.info(
            f"Image node(284): {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_FOUND')}"
        )
    else:
        logger.info(
            f"Video node(228): {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_FOUND')}"
        )
    logger.info(
        f"Audio node(125): {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
    )
    logger.info(
        f"Text node(241): {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_FOUND')}"
    )
    if person_count == "multi":
        if "307" in prompt:
            logger.info(
                f"Second audio node(307): {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )
        elif "313" in prompt:
            logger.info(
                f"Second audio node(313): {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt queued successfully: {result}")
        return result
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        logger.error(f"Response body: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        logger.error(f"Error queueing prompt: {e}")
        raise


def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def get_videos(ws, prompt, input_type="image", person_count="single", job=None):
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    logger.info(f"Workflow execution started: prompt_id={prompt_id}")

    # Background heartbeat to prevent stale job requeue during silent phases
    # (model loading, first sampling step) where ComfyUI sends no WebSocket messages
    heartbeat_stop = threading.Event()

    def heartbeat():
        hb = None
        try:
            from runpod.serverless.modules.rp_ping import Heartbeat
            hb = Heartbeat()
            logger.info("Direct sidecar heartbeat initialized")
        except Exception as e:
            logger.warning(f"Could not init direct heartbeat: {e}")

        while not heartbeat_stop.wait(8):
            # Direct sidecar ping (may prevent stale requeue)
            if hb:
                try:
                    hb._send_ping()
                except Exception as e:
                    logger.warning(f"Sidecar ping failed: {e}")
            # Client-visible progress (does NOT prevent requeue)
            if job:
                runpod.serverless.progress_update(job, "Processing...")

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    output_videos = {}
    try:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                logger.error("WebSocket recv timed out (120s) — ComfyUI may be unresponsive")
                raise Exception("WebSocket timeout: no message from ComfyUI for 120 seconds")
            except (websocket.WebSocketConnectionClosedException, ConnectionError) as e:
                logger.error(f"WebSocket disconnected: {e}")
                raise Exception(f"WebSocket disconnected: {e}")
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is not None:
                        logger.info(f"Executing node: {data['node']}")
                        if job:
                            runpod.serverless.progress_update(job, f"Executing node: {data['node']}")
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        logger.info("Workflow execution complete")
                        break
                elif message["type"] == "progress":
                    data = message["data"]
                    if job:
                        runpod.serverless.progress_update(
                            job, f"Node {data.get('node', '?')}: {data['value']}/{data['max']}"
                        )
                elif message["type"] == "execution_error":
                    error_data = message.get("data", {})
                    error_msg = f"ComfyUI execution error: {error_data}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                continue
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)

    logger.info(f"Fetching history: prompt_id={prompt_id}")
    history = get_history(prompt_id)[prompt_id]
    logger.info(f"Output node count: {len(history['outputs'])}")

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        videos_output = []
        if "gifs" in node_output:
            logger.info(f"Node {node_id}: found {len(node_output['gifs'])} video(s)")
            for idx, video in enumerate(node_output["gifs"]):
                video_path = video["fullpath"]
                logger.info(f"Video file path: {video_path}")

                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    logger.info(f"Video {idx+1}: {video_path} ({file_size} bytes)")
                else:
                    logger.warning(f"Video file does not exist: {video_path}")

                videos_output.append(video_path)
        else:
            logger.info(f"Node {node_id}: no video output")
        output_videos[node_id] = videos_output

    logger.info(f"Collected video paths from {len(output_videos)} node(s)")
    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)


def get_workflow_path(input_type, person_count):
    """Return the workflow JSON path for the given input_type and person_count."""
    if input_type == "image":
        if person_count == "single":
            return "/I2V_single.json"
        else:  # multi
            return "/I2V_multi.json"
    else:  # video
        if person_count == "single":
            return "/V2V_single.json"
        else:  # multi
            return "/V2V_multi.json"


def get_audio_duration(audio_path):
    """Return audio file duration in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"Failed to get audio duration ({audio_path}): {e}")
        return None


def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """Calculate max_frames from audio duration."""
    durations = []

    duration1 = get_audio_duration(wav_path)
    if duration1 is not None:
        durations.append(duration1)
        logger.info(f"Audio 1 duration: {duration1:.2f}s")

    if wav_path_2:
        duration2 = get_audio_duration(wav_path_2)
        if duration2 is not None:
            durations.append(duration2)
            logger.info(f"Audio 2 duration: {duration2:.2f}s")

    if not durations:
        logger.warning("Could not calculate audio duration, using default max_frames=81")
        return 81

    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81

    logger.info(f"Max audio duration: {max_duration:.2f}s, calculated max_frames: {max_frames}")
    return max_frames


def handler(job):
    logger.info(f"RunPod SDK v{runpod.__version__}")
    logger.info(f"RUNPOD_WEBHOOK_PING: {'SET' if os.environ.get('RUNPOD_WEBHOOK_PING') else 'NOT SET'}")
    logger.info(f"RUNPOD_PING_INTERVAL: {os.environ.get('RUNPOD_PING_INTERVAL', '10000 (default)')}")

    job_input = job.get("input", {})
    use_network_volume = job_input.get("network_volume", False)
    request_key = get_request_key(job)

    log_input = job_input.copy()
    for key in ["image_base64", "video_base64", "wav_base64", "wav_base64_2"]:
        if key in log_input:
            log_input[key] = truncate_base64_for_log(log_input[key])

    logger.info(f"Received job input: {log_input}")
    if request_key:
        logger.info(f"Request key: {request_key}")
        cached_result = load_cached_result(request_key, use_network_volume)
        if cached_result is not None:
            logger.info(f"Idempotent cache hit for request_key={request_key}")
            return cached_result

    task_id = f"task_{uuid.uuid4()}"

    input_type = job_input.get("input_type", "image")
    person_count = job_input.get("person_count", "single")

    logger.info(f"Workflow type: {input_type}, person_count: {person_count}")

    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"Using workflow: {workflow_path}")

    media_path = None
    if input_type == "image":
        if "image_path" in job_input:
            media_path = process_input(
                job_input["image_path"], task_id, "input_image.jpg", "path"
            )
        elif "image_url" in job_input:
            media_path = process_input(
                job_input["image_url"], task_id, "input_image.jpg", "url"
            )
        elif "image_base64" in job_input:
            media_path = process_input(
                job_input["image_base64"], task_id, "input_image.jpg", "base64"
            )
        else:
            media_path = "/examples/image.jpg"
            logger.info("Using default image: /examples/image.jpg")
    else:  # video
        if "video_path" in job_input:
            media_path = process_input(
                job_input["video_path"], task_id, "input_video.mp4", "path"
            )
        elif "video_url" in job_input:
            media_path = process_input(
                job_input["video_url"], task_id, "input_video.mp4", "url"
            )
        elif "video_base64" in job_input:
            media_path = process_input(
                job_input["video_base64"], task_id, "input_video.mp4", "base64"
            )
        else:
            media_path = "/examples/image.jpg"
            logger.info("Using default image: /examples/image.jpg")

    wav_path = None
    wav_path_2 = None

    if "wav_path" in job_input:
        wav_path = process_input(
            job_input["wav_path"], task_id, "input_audio.wav", "path"
        )
    elif "wav_url" in job_input:
        wav_path = process_input(
            job_input["wav_url"], task_id, "input_audio.wav", "url"
        )
    elif "wav_base64" in job_input:
        wav_path = process_input(
            job_input["wav_base64"], task_id, "input_audio.wav", "base64"
        )
    else:
        wav_path = "/examples/audio.mp3"
        logger.info("Using default audio: /examples/audio.mp3")

    if person_count == "multi":
        if "wav_path_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_path_2"], task_id, "input_audio_2.wav", "path"
            )
        elif "wav_url_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_url_2"], task_id, "input_audio_2.wav", "url"
            )
        elif "wav_base64_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_base64_2"], task_id, "input_audio_2.wav", "base64"
            )
        else:
            wav_path_2 = wav_path
            logger.info("No second audio provided, reusing first audio")

    prompt_text = job_input.get("prompt", "A person talking naturally")
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)

    max_frame = job_input.get("max_frame")
    if max_frame is None:
        logger.info("max_frame not provided, auto-calculating from audio duration")
        max_frame = calculate_max_frames_from_audio(
            wav_path, wav_path_2 if person_count == "multi" else None
        )
    else:
        logger.info(f"User-specified max_frame: {max_frame}")

    logger.info(f"Workflow config: prompt='{prompt_text}', width={width}, height={height}, max_frame={max_frame}")
    logger.info(f"Media path: {media_path}")
    logger.info(f"Audio path: {wav_path}")
    if person_count == "multi":
        logger.info(f"Second audio path: {wav_path_2}")

    prompt = load_workflow(workflow_path)

    force_offload = job_input.get("force_offload", True)
    logger.info(f"Config: force_offload={force_offload}")

    sampler_node_id = None
    preferred_id = "128"
    if preferred_id in prompt and prompt[preferred_id].get("class_type") == "WanVideoSampler":
        sampler_node_id = preferred_id
    else:
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "WanVideoSampler":
                sampler_node_id = node_id
                break

    if sampler_node_id:
        inputs = prompt[sampler_node_id].setdefault("inputs", {})
        inputs["force_offload"] = force_offload
        logger.info(f"Node {sampler_node_id} (WanVideoSampler) updated: force_offload={force_offload}")
    else:
        logger.warning("WanVideoSampler node not found, using workflow defaults")

    if not os.path.exists(media_path):
        logger.error(f"Media file does not exist: {media_path}")
        return {"error": f"Media file not found: {media_path}"}

    if not os.path.exists(wav_path):
        logger.error(f"Audio file does not exist: {wav_path}")
        return {"error": f"Audio file not found: {wav_path}"}

    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        logger.error(f"Second audio file does not exist: {wav_path_2}")
        return {"error": f"Second audio file not found: {wav_path_2}"}

    logger.info(f"Media file size: {os.path.getsize(media_path)} bytes")
    logger.info(f"Audio file size: {os.path.getsize(wav_path)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"Second audio file size: {os.path.getsize(wav_path_2)} bytes")

    if input_type == "image":
        prompt["284"]["inputs"]["image"] = media_path
    else:
        prompt["228"]["inputs"]["video"] = media_path

    prompt["125"]["inputs"]["audio"] = wav_path
    prompt["241"]["inputs"]["positive_prompt"] = prompt_text
    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height

    prompt["270"]["inputs"]["value"] = max_frame
    prompt["192"]["inputs"]["frame_window_size"] = max_frame

    if person_count == "multi":
        if input_type == "image":
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_path_2
        else:
            if "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_path_2

    runpod.serverless.progress_update(job, "Connecting to ComfyUI...")
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request

            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection successful (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI server")
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            ws.settimeout(120)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timed out (3 min)")
            time.sleep(5)
    videos = get_videos(ws, prompt, input_type, person_count, job=job)
    ws.close()
    logger.info("WebSocket closed")

    output_video_path = None
    logger.info("Searching for output video...")

    for node_id in videos:
        if videos[node_id]:
            output_video_path = videos[node_id][0]
            logger.info(f"Found output video in node {node_id}: {output_video_path}")
            break
        else:
            logger.info(f"Node {node_id} is empty")

    if not output_video_path:
        logger.error("No output video found, all nodes are empty")
        return {"error": "No video output found"}

    if not os.path.exists(output_video_path):
        logger.error(f"Output video file does not exist: {output_video_path}")
        return {"error": f"Video file not found: {output_video_path}"}

    save_cached_video(request_key, output_video_path)
    logger.info(f"Use network volume: {use_network_volume}")

    if use_network_volume and os.path.isdir("/runpod-volume"):
        logger.info("Copying video to network volume")
        try:
            output_suffix = request_key or task_id
            output_filename = f"infinitetalk_{output_suffix}.mp4"
            output_path = f"/runpod-volume/{output_filename}"
            logger.info(f"Source: {output_video_path}")
            logger.info(f"Destination: {output_path}")

            source_file_size = os.path.getsize(output_video_path)
            logger.info(f"Source file size: {source_file_size} bytes")

            shutil.copy2(output_video_path, output_path)
            logger.info("File copy complete")

            copied_file_size = os.path.getsize(output_path)
            logger.info(f"Copied file size: {copied_file_size} bytes")

            if source_file_size == copied_file_size:
                logger.info(f"Video copied to '{output_path}'")
            else:
                logger.warning(f"File size mismatch: source={source_file_size}, copy={copied_file_size}")

            return {"video_path": output_path}

        except Exception as e:
            logger.error(f"Video copy failed: {e}")
            return {"error": f"Video copy failed: {e}"}
    elif use_network_volume:
        logger.warning("network_volume requested but /runpod-volume is unavailable; using base64")

    runpod.serverless.progress_update(job, "Encoding video output...")
    logger.info("Starting base64 encoding")
    logger.info(f"Video file path: {output_video_path}")

    try:
        file_size = os.path.getsize(output_video_path)
        logger.info(f"Source file size: {file_size} bytes")

        with open(output_video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")

        encoded_size = len(video_data)
        logger.info(f"Base64 encoding complete: {encoded_size} chars")
        logger.info(f"Returning base64 video: {truncate_base64_for_log(video_data)}")
        return {"video": video_data}

    except Exception as e:
        logger.error(f"Base64 encoding failed: {e}")
        return {"error": f"Base64 encoding failed: {e}"}


runpod.serverless.start({"handler": handler})
