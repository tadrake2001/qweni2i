import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

def save_data_if_base64(data_input, temp_dir, output_filename):
    """입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다."""
    if not isinstance(data_input, str):
        return data_input

    try:
        decoded_data = base64.b64decode(data_input)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        print(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError):
        print(f"➡️ '{data_input}'은(는) 파일 경로로 처리합니다.")
        return data_input

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req).read())

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
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

# i2i.json workflow node IDs
_WORKFLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow", "i2i.json")

# Node IDs from i2i.json
_NODE_LOAD_IMAGE = "437"           # LoadImage - input image
_NODE_TEXT_GENERATE = "7"          # TextGenerate - receives user prompt
_NODE_SEED = "438:3"               # KSampler - seed, steps, cfg, etc.
_NODE_SAVE_IMAGE = "436"           # SaveImage - output

def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")

    # Workflow path
    if not os.path.exists(_WORKFLOW_PATH):
        return {"error": f"Workflow file not found: {_WORKFLOW_PATH}"}

    prompt = load_workflow(_WORKFLOW_PATH)

    # ------------------------------
    # Image input handling
    # ------------------------------
    # ComfyUI loads images from its input directory
    COMFYUI_INPUT_DIR = "/ComfyUI/input"
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)

    image_path = None
    if "image_base64" in job_input:
        task_id = f"task_{uuid.uuid4().hex}"
        filename = f"{task_id}_input.jpg"
        abs_path = os.path.join(COMFYUI_INPUT_DIR, filename)
        image_path = save_data_if_base64(job_input["image_base64"], COMFYUI_INPUT_DIR, filename)
        # Use only the filename (relative to ComfyUI input dir)
        image_path = filename
    elif "image_path" in job_input:
        image_path = job_input["image_path"]
    elif "image_url" in job_input:
        task_id = f"task_{uuid.uuid4().hex}"
        filename = f"{task_id}_input.jpg"
        abs_path = os.path.join(COMFYUI_INPUT_DIR, filename)
        download_file_from_url(job_input["image_url"], abs_path)
        image_path = filename
    else:
        return {"error": "Image required (image_base64 / image_path / image_url)"}

    # Set image to LoadImage node
    prompt[_NODE_LOAD_IMAGE]["inputs"]["image"] = image_path

    # ------------------------------
    # Prompt handling
    # ------------------------------
    user_prompt = job_input.get("prompt", "")
    # User prompt goes to TextGenerate (7), which outputs to PrimitiveStringMultiline (439)
    prompt[_NODE_TEXT_GENERATE]["inputs"]["prompt"] = user_prompt

    # ------------------------------
    # Seed handling
    # ------------------------------
    if "seed" in job_input:
        prompt[_NODE_SEED]["inputs"]["seed"] = job_input["seed"]
    if "steps" in job_input:
        prompt[_NODE_SEED]["inputs"]["steps"] = job_input["steps"]
    if "cfg" in job_input:
        prompt[_NODE_SEED]["inputs"]["cfg"] = job_input["cfg"]

    # ------------------------------
    # Connect to ComfyUI
    # ------------------------------
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    http_url = f"http://{server_address}:8188/"
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection success (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI server")
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180/5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connection success (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timeout (3 min)")
            time.sleep(5)

    images = get_images(ws, prompt)
    ws.close()

    if not images:
        return {"error": "No images generated"}

    # Return image from SaveImage node (436)
    if _NODE_SAVE_IMAGE in images and images[_NODE_SAVE_IMAGE]:
        return {"image": images[_NODE_SAVE_IMAGE][0]}

    # Fallback: return first image found
    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}

    return {"error": "No output images found"}

def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget failed: {result.stderr}")
            raise Exception(f"Download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timeout")
        raise Exception("Download timeout")
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise Exception(f"Download error: {e}")

runpod.serverless.start({"handler": handler})