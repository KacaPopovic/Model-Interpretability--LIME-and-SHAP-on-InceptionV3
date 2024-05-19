import json
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap
from tensorflow.keras.preprocessing import image
import numpy as np


def f(x):
    tmp = x.copy()
    tmp = preprocess_input(tmp)
    tmp = tmp[None, :] if len(tmp.shape) == 3 else tmp  # expand dimensions if necessary
    return model(tmp)

if __name__ == "__main__":

    # load pre-trained model and data
    model = ResNet50(weights="imagenet")
    images = ["image1.jpg", "image2.jpg"]

    loaded_images = []
    for img_path in images:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        loaded_images.append(img_array)

    # stack images into a single numpy array
    X = np.vstack(loaded_images)
    X, y = shap.datasets.imagenet50()
    # getting ImageNet 1000 class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with open(shap.datasets.cache(url)) as file:
        class_names = [v[1] for v in json.load(file).values()]
    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    # create an explainer with model and image masker
    explainer = shap.Explainer(f, masker, output_names=class_names)
    # calling explainer on preprocessed images
    shap_values = explainer(
        X, max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:4]
    )
    # output with shap values
    shap.image_plot(shap_values)