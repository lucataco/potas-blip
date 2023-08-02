from potassium import Potassium, Request, Response
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


from BLIP import models
from BLIP.models.blip import blip_decoder
from BLIP.models.blip_vqa import blip_vqa
from BLIP.models.blip_itm import blip_itm

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    image_captioning = blip_decoder(
        pretrained="checkpoints/model*_base_caption.pth",
        image_size=384,
        vit="base",
    )
    visual_question_answering = blip_vqa(
        pretrained="checkpoints/model*_vqa.pth",
        image_size=480,
        vit="base"
    )
    image_text_matching = blip_itm(
        pretrained="checkpoints/model_base_retrieval_coco.pth",
        image_size=384,
        vit="base",
    )
    context = {
        "image_captioning": image_captioning,
        "question_answering": visual_question_answering,
        "text_matching": image_text_matching
    }
    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    # Get inputs
    image = request.json.get("image")
    task = request.json.get("task")
    question = request.json.get("question")
    caption = request.json.get("caption")

    # Load models
    image_captioning = context.get("image_captioning")
    visual_question_answering = context.get("question_answering")
    text_matching = context.get("text_matching")

    im = load_image(
        image,
        image_size=480 if task == "visual_question_answering" else 384,
        device="cuda",
    )
    
    output = ''
    if task == "image_captioning":
        image_captioning.eval()
        model = image_captioning.to("cuda")
        with torch.no_grad():
            caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
            output = "Caption: " + caption[0]
    elif task == "visual_question_answering":
        visual_question_answering.eval()
        model = visual_question_answering.to("cuda")
        with torch.no_grad():
            answer = model(im, str(question), train=False, inference="generate")
            output = "Answer: " + answer[0]
    else:
        # image_text_matching
        text_matching.eval()
        model = text_matching.to("cuda")
        itm_output = model(im, caption, match_head="itm")
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
        itc_score = model(im, caption, match_head="itc")
        output = f"The image and text is matched with a probability of {itm_score.item():.4f}.\n The image feature and text feature has a cosine similarity of {itc_score.item():.4f}."
    
    return Response(
        json = {"output": output}, 
        status=200
    )

def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

if __name__ == "__main__":
    app.serve()
