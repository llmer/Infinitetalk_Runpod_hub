#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Model provisioning ---
# Models are no longer baked into the Docker image to keep it small (~22GB vs ~55GB).
# On first boot with a network volume, models download once and persist across restarts.
MODEL_DIR="/ComfyUI/models"
VOLUME_DIR="/runpod-volume"
VOLUME_MODEL_DIR="$VOLUME_DIR/models"
SENTINEL=".models_ready"

download_models() {
    local dest="$1"
    echo "Downloading models to $dest ..."
    mkdir -p "$dest/diffusion_models" "$dest/loras" "$dest/vae" \
             "$dest/text_encoders" "$dest/clip_vision" \
             "$dest/transformers/TencentGameMate"

    # Parallel wget downloads (~33GB total)
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors \
         -O "$dest/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors \
         -O "$dest/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors \
         -O "$dest/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
         -O "$dest/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
         -O "$dest/vae/Wan2_1_VAE_bf16.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors \
         -O "$dest/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors" &
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
         -O "$dest/clip_vision/clip_vision_h.safetensors" &
    wget -q https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors \
         -O "$dest/diffusion_models/MelBandRoformer_fp16.safetensors" &
    wait

    # Verify all downloads succeeded
    for f in \
        "$dest/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
        "$dest/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" \
        "$dest/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
        "$dest/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
        "$dest/vae/Wan2_1_VAE_bf16.safetensors" \
        "$dest/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
        "$dest/clip_vision/clip_vision_h.safetensors" \
        "$dest/diffusion_models/MelBandRoformer_fp16.safetensors"; do
        [ -s "$f" ] || { echo "FAILED: $f is missing or empty"; exit 1; }
    done

    # wav2vec2 model via huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TencentGameMate/chinese-wav2vec2-base',
    local_dir='$dest/transformers/TencentGameMate/chinese-wav2vec2-base',
    revision='main',
)
"
    [ -s "$dest/transformers/TencentGameMate/chinese-wav2vec2-base/pytorch_model.bin" ] || \
        { echo "FAILED: wav2vec2 model download missing"; exit 1; }

    touch "$dest/$SENTINEL"
    echo "Model download complete."
}

if [ -d "$VOLUME_DIR" ]; then
    echo "Network volume detected at $VOLUME_DIR"
    if [ -f "$VOLUME_MODEL_DIR/$SENTINEL" ]; then
        echo "Models already present on network volume."
    else
        download_models "$VOLUME_MODEL_DIR"
    fi
    # Symlink each model subdir from volume into ComfyUI
    for subdir in diffusion_models loras vae text_encoders clip_vision transformers; do
        rm -rf "$MODEL_DIR/$subdir"
        ln -sf "$VOLUME_MODEL_DIR/$subdir" "$MODEL_DIR/$subdir"
    done
    echo "Model symlinks created."
else
    echo "WARNING: No network volume detected. Models will download to local disk."
    echo "WARNING: This adds ~3-5 min to every cold start. Attach a network volume for persistent storage."
    if [ ! -f "$MODEL_DIR/$SENTINEL" ]; then
        download_models "$MODEL_DIR"
    fi
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground
echo "Starting the handler..."
exec python handler.py
