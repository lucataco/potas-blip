import banana_dev as client
from io import BytesIO
from PIL import Image
import base64
import time

# Localhost test
my_model = client.Client(
    api_key="",
    model_key="",
    url="http://localhost:8000",
)

# inputs = {
#     "image": "demo.jpg",
#     "task": "image_captioning"
# }

# inputs = {
#     "image": "demo.jpg",
#     "task": "image_text_matching",
#     "caption": "a dog and a women are sitting at the beach"
# }

# inputs = {
#     "image": "demo.jpg",
#     "task": "image_text_matching",
#     "caption": "a dog and a cat are playing in the garden"
# }

inputs = {
    "image": "demo.jpg",
    "task": "visual_question_answering",
    "question": "where is the woman?"
}


# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first
# method argument ("/")to specify a
# different route.
t1 = time.time()
result, meta = my_model.call("/", inputs)
t2 = time.time()

output = result["output"]
print(output)
print("Time to run: ", t2 - t1)