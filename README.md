# Qwen Image Edit for RunPod Serverless

This project deploys a Qwen Image Edit workflow (via ComfyUI) as a RunPod Serverless API.

## Key Features

*   **Prompt-Guided Image Editing**: Edit images based on a text prompt using Qwen's image editing model.
*   **Flexible Inputs**: Provide images via URL, file path, or Base64 string.
*   **ComfyUI-based**: Built on ComfyUI for reliable workflow execution.

## Files

*   **Dockerfile**: Environment setup with ComfyUI, custom nodes (ComfyUI-Qwen-Image-Edit, KJNodes), and models.
*   **handler_i2i.py**: Serverless handler for i2i.json workflow.
*   **entrypoint.sh**: Worker initialization and ComfyUI startup.
*   **workflow/i2i.json**: ComfyUI workflow for single-image editing.

## Input

The `input` object accepts the following fields:

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `prompt` | `string` | Yes | - | Text prompt guiding the edit. |
| `image_url` or `image_base64` or `image_path` | `string` | Yes | - | Input image (URL/Base64/path). |
| `seed` | `integer` | No | random | Random seed for reproducible output. |
| `steps` | `integer` | No | 4 | Sampling steps. |
| `cfg` | `float` | No | 1.0 | CFG scale. |

**Request Example:**

```json
{
  "input": {
    "prompt": "add watercolor style, soft pastel tones",
    "image_url": "https://example.com/your-image.jpg",
    "seed": 12345,
    "steps": 4,
    "cfg": 1.0
  }
}
```

## Output

#### Success

Returns a JSON object with the generated image (Base64-encoded).

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}
```

#### Error

```json
{
  "error": "Error message"
}
```

## Testing

Set `runpod_API_KEY` and `qwen_image_edit` (endpoint ID) in `test.env`, then:

```bash
# Test with URL
python test_api.py --mode url --image-url "https://example.com/your-image.jpg"

# Test with Base64
python test_api.py --mode base64

# Test with JSON file
python test_api.py --json example_request.json
```

## Workflow

The i2i.json workflow uses these ComfyUI nodes:

| Node ID | Type | Purpose |
| --- | --- | --- |
| 437 | LoadImage | Input image |
| 7 | TextGenerate | Process prompt |
| 438:3 | KSampler | Sampling (seed, steps, cfg) |
| 436 | SaveImage | Output image |

## Models

Downloaded from HuggingFace:

*   **Diffusion**: `Qwen-Image-Edit_ComfyUI` (qwen_image_edit_2509_fp8_e4m3fn.safetensors)
*   **LoRA**: `Qwen-Image-Lightning` (Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors)
*   **Text Encoders**: `flux2-klein` (qwen_3_4b.safetensors), `Qwen-Image_ComfyUI` (qwen_2.5_vl_7b_fp8_scaled.safetensors)
*   **VAE**: `Qwen-Image_ComfyUI` (qwen_image_vae.safetensors)

## License

Based on ComfyUI and Qwen projects.