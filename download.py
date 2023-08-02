# This file runs during container build time to get model weights built into the container
import os

def download_model():
    # Github repo
    if not os.path.exists('/src'):
        os.system('git clone https://github.com/salesforce/BLIP /src')
    # Models
    os.makedirs('/src/checkpoints/', exist_ok=True)
    # download weights
    if not os.path.exists('/src/checkpoints/model*_base_caption.pth'):
        os.system('wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth -P /src/checkpoints')
    if not os.path.exists('checkpoints/model*_vqa.pth'):
        os.system('wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth -P /src/checkpoints')
    if not os.path.exists('checkpoints/model_base_retrieval_coco.pth'):
        os.system('wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth -P /src/checkpoints')


if __name__ == "__main__":
    download_model()