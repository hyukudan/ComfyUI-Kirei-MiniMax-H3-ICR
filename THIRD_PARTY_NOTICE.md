# Third-party notices

## MiniMax H3 latent-upscaler checkpoint

The native Kirei runtime can load, and can explicitly download when requested, the following
separately distributed checkpoint:

- Project: `LBH-123-AI/Minimax_h3_latent_Upscaler`
- Model: `minimax_h3_latent_upscaler_3d_bf16.safetensors`
- Source: <https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler>
- License declared by the model card: Apache-2.0
- SHA-256: `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`
- Expected size: `690592992` bytes

The checkpoint is not committed to or redistributed inside this Git repository. Downloading it
remains an explicit user action. The Kirei node verifies both size and SHA-256 before making the
file visible to ComfyUI.

The license and training-provenance statements above are those published by the model author. Users
should also review the terms governing MiniMax H3 and any training sources or derived weights; this
notice does not independently certify that provenance.

The model card reports approximately 70,000 video pairs and 8,000 2K image pairs, including 1.5x
and continuous 1x-4x scale training. Kirei has not independently audited those datasets or claims.

MiniMax H3 and any other model, LoRA, VAE or text-encoder weights remain separate dependencies and
retain their respective licenses.
