import requests
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2, preprocess_input, decode_predictions

class ImageClassifier:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
    
    def classify_image(self, image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image), axis=0)
        image_array = preprocess_input(image_array)
        
        predictions = self.model.predict(image_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0]

        return decoded_predictions[0][1]

def main():
    image_classifier = ImageClassifier()

    # Replace the image_url with the actual URL of the image you want to classify
    image_url = 'https://example.com/path/to/your/image.jpg'
    result = image_classifier.classify_image(image_url)

    print(f'The image is classified as: {result}')

if __name__ == '__main__':
    main()
