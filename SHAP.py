import json
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap
from tensorflow.keras.preprocessing import image

# Declare model as global variable to make it accessible within function f()
global model


def f(x):
    # Access global variable model within the function
    global model
    tmp = x.copy()
    tmp = preprocess_input(tmp)
    tmp = tmp[None, :] if len(tmp.shape) == 3 else tmp
    return model(tmp)


if __name__ == "__main__":
    model = ResNet50(weights="imagenet")
    images = ["image1.jpg", "image2.jpg"]
    loaded_images = []
    for img_path in images:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        loaded_images.append(img_array)

    X = np.vstack(loaded_images)
    predictions = f(X)

    try:
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with open(shap.datasets.cache(url)) as file:
            class_names = [v[1] for v in json.load(file).values()]
    except Exception as e:
        print('Error:', e)

    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    explainer = shap.Explainer(f, masker, output_names=class_names)
    shap_values = explainer(X, max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
    shap.image_plot(shap_values)