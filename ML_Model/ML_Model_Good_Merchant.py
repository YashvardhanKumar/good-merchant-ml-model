import requests
from io import BytesIO
from PIL import Image
import sys
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img
import numpy as np



def process_image_url(path):
    print("[INFO] loading and preprocessing image...")

    # path = "https://5.imimg.com/data5/NP/AO/MY-34422787/image-1-500x500.jpg"

    response = requests.get(path)
    img_bytes = BytesIO(response.content)
    image = Image.open(img_bytes)
    image = image.convert('RGB')
    image = image.resize((256, 256), Image.NEAREST)
    return image


def process_image_binary(path):
    # path = 'sample_test/41.jpg'
    image = load_img(path, target_size=(256, 256))
    return image


def predict_image(image, model):
    
    products = ['earpods', 'headphone', 'laptop',
                'mobile', 'neckband', 'powerbank', 'tablet']
    
    vgg16 = keras.applications.vgg16.VGG16(
        include_top=False, weights='imagenet')
    
    sys.modules['Image'] = Image
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.

    bt_prediction = vgg16.predict(image)
    preds = model.predict(bt_prediction)

    for i, product, x in zip(range(0, 6), products, preds[0]):
        print("ID:", i, "Label:", product, round(x*100, 2), "%")

    print("Max prediction:")

    class_predicted = np.argmax(preds, axis=1)

    print("ID: ", class_predicted[0], "Label: ", products[class_predicted[0]])
    return products[class_predicted[0]]
