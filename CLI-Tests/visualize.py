import requests
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load your image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_download = requests.get(image_url)
image = Image.open(BytesIO(image_download.content)).convert('RGB')
if image is None:
    raise FileNotFoundError("The image file was not found.")

# first coordinates output from COGVLM -- x1, y1, x2, y2
x1, y1, x2, y2 = 546, 60, 998, 775

# Draw the bounding box on the image
draw = ImageDraw.Draw(image)
draw.rectangle([x1, y1, x2, y2], outline ="green", width=4)

# Display the image using matplotlib
plt.imshow(image)
plt.title('Image with Bounding Box')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
