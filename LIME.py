import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.linear_model import LinearRegression
import json

class Lime:
    """
    This class implements the Lime algorithm for explainable machine learning.

    Parameters:
        model: The trained machine learning model.

    Attributes:
        model: The trained machine learning model.
        image: The input image for explanation.
        segments: The segmented image.
        perturbed_images: List of perturbed images.
        perturbations: List of perturbations for each perturbed image.
        predictions: List of predicted probabilities for each perturbed image.
        similarities: List of similarity weights for each perturbation.
        original_prediction: The original prediction for the input image.
        class_names: Dictionary of class names loaded from 'imagenet_class_index.json'.

    Methods:
        __init__: Initializes an instance of the Lime class.
        load_directory: Loads the class names from a JSON file.
        load_image: Loads an image for explanation.
        delete_memory: Deletes all the attributes of the class.
        generate_superpixels: Generates superpixels for the segmented image.
        create_perturbed_images_and_predict: Creates perturbed images and predicts probabilities.
        plot_image: Plots the provided image tensor.
        perform_predictions: Predicts probabilities for all perturbed images.
        fit_regression: Fits a linear regression model to explain the predictions.
        visualize_impact: Visualizes the impact of perturbations on the prediction.
        get_top_predictions: Gets the top N predictions from the model output.
    """

    def __init__(self, model):
        self.model = model
        self.image = None
        self.segments = None
        self.perturbed_images = []
        self.perturbations = []
        self.predictions = []
        self.similarities = []
        self.original_prediction = None
        self.class_names = self.load_directory('imagenet_class_index.json')

    def load_directory(self, json_path):
        """
        Loads a JSON file from the specified directory and converts it into a dictionary where keys are indices and values are class names.

        :param json_path: The path to the JSON file to be loaded
        :return: A dictionary where keys are indices and values are class names
        """
        with open(json_path) as f:
            class_idx = json.load(f)

        # Convert the JSON file to a dictionary where keys are indices and values are class names
        class_names = {int(key): value[1] for key, value in class_idx.items()}
        return class_names
    def load_image(self, image_path):
        """
        Load an image and preprocess it for input to the model.

        :param image_path: The path to the image file.
        :return: None

        This method loads the image from the specified `image_path` and converts it to the RGB format. It then applies a series of transformations to preprocess the image for input to the model. The transformations include resizing the image to 299x299 pixels, center cropping the image, converting it to a tensor, and normalizing it using the specified mean and standard deviation values.

        After preprocessing the image, the method assigns the preprocessed image to the `self.image` attribute. It then feeds the image to the model and obtains the output. The output is processed to obtain the top predictions, and the corresponding class name is extracted from the `self.class_names` list. Finally, the method calls the `self.plot_image` method to display the preprocessed image along with its predicted class name.

        Note: This method assumes that the necessary imports and variable declarations have been made before calling the method.

        Example usage:
        ```python
        load_image('/path/to/image.jpg')
        ```
        """
        img = Image.open(image_path).convert('RGB')
        # Convert in a format for Inception V3
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image = preprocess(img).unsqueeze(0)
        output = self.model(self.image)
        output = self.get_top_predictions(output)
        class_name = self.class_names[output[1]]
        self.plot_image(self.image, class_name)

    def delete_memory(self):
        """
        Delete the memory stored in the object.

        :return: None
        """
        self.image = None
        self.segments = None
        self.perturbed_images = []
        self.perturbations = []
        self.predictions = []
        self.similarities = []
        self.original_prediction = None

    def generate_superpixels(self, n_segments=50):
        """
        :param n_segments: The number of desired superpixels to generate. Default value is 50.
        :return: None

        This method generates superpixels for the given image using the SLIC algorithm. The generated superpixels are stored in the 'segments' attribute of the object.
        """
        image = self.image
        image_np = image.squeeze().permute(1, 2, 0).numpy()
        image_np = img_as_float(image_np)
        segments = slic(image_np, n_segments=n_segments, compactness=10, sigma=1)  # Simple linear iterative clustering
        self.segments = segments


    def create_perturbed_images_and_predict(self, num_pertubations=300):
        """
        Creates perturbed images and predicts their class probabilities.

        :param num_pertubations: The number of perturbations to create. Default is 300.
        :return: None
        """
        unique_segments = np.unique(self.segments)
        perturbed_images = []

        for i in range(num_pertubations):
            decisions = np.random.choice([0, 1], size=len(unique_segments), p=[0.5, 0.5])
            # Create a mask for all segments in one go
            mask = np.isin(self.segments, unique_segments[decisions == 1])

            # Create perturbed image using the mask
            perturbed_image = self.image * torch.tensor(mask, dtype=torch.float32)
            self.perturbed_images.append(perturbed_image)
            perturbed_images.append(perturbed_image)
            self.perturbations.append(decisions)
            weight = torch.nn.functional.cosine_similarity(
                self.image.flatten(),
                perturbed_image.flatten(), dim=0)
            self.similarities.append(weight)

        self.original_prediction = self.model(self.image)
        top_classes_original = self.get_top_predictions(self.original_prediction)
        for img_tensor in perturbed_images:
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            perturbed_image_probs = probs.squeeze()[top_classes_original].tolist()
            # self.plot_image(img_tensor)
            self.predictions.append(perturbed_image_probs)



    def plot_image(self, image_tensor, title=None):
        """
        Plots the provided image tensor.

        image_tensor: PyTorch tensor representing the image to be plotted
        title: Optional, title for the plot
        """

        # Remove the batch dimension if present
        image = image_tensor.squeeze().numpy()

        # If image is RGB, transpose the color channel to the last dimension
        if len(image.shape) == 3:
            image = image.transpose((1, 2, 0))

        # Normalize to [0, 1] range for correct display
        image = (image - image.min()) / (image.max() - image.min())

        # Plot the image
        plt.imshow(image)
        plt.axis("off")  # Remove axis
        if title is not None:
            plt.title(title)
        plt.show()

    def perform_predictions(self):
        """
        Perform predictions using the model on the given image and perturbed images.

        :return: None
        """
        self.original_prediction = self.model(self.image)
        for img_tensor in self.perturbed_images:
            output = self.model(img_tensor)
            predictions = self.get_top_predictions(output)
            self.predictions.append(predictions)

    def fit_regression(self):
        """
        Fits a linear regression model to the provided data.

        :return: The coefficients of the regression model.
        """
        # Assign instance variables to local variables
        X = np.array(self.perturbations)  # Feature variable
        y = np.array(self.predictions)  # Target variable
        weights = np.array(self.similarities)  # Sample weights

        # Initialize the model
        # You may need to reshape the arrays as per the model's requirement
        self.regression_model = LinearRegression()

        # Fit the model
        self.regression_model.fit(X, y, sample_weight=weights)
        return self.regression_model.coef_

    def visualize_impact(self, prediction_index=1, num_superpixels=4):
        """
        :param prediction_index: The index of the prediction for which the impact is visualized. By default, it is set to 1.
        :param num_superpixels: The number of top superpixels to consider. By default, it is set to 4.
        :return: None

        This method visualizes the impact of each superpixel on a specific prediction. It creates a mask of ones with the size equal to the number of unique segments in the image. Then, it selects the top `num_superpixels` superpixels based on the coefficient values from the regression model for the specified `prediction_index`. The selected superpixels are set to zero in the mask, creating a perturbed image. It then passes the perturbed image through a model and retrieves the class prediction for the specified `prediction_index`. Finally, it calls the `plot_image` method to display the perturbed image with the corresponding class name.

        """
        coeffs = self.regression_model.coef_[prediction_index,:]
        unique_segments = np.unique(self.segments)
        sorted_indices = np.argsort(-self.regression_model.coef_[prediction_index, :])
        highest_indices = sorted_indices[0:num_superpixels]

        # Creating a mask of ones, size as 'unique_segments'
        decisions = np.zeros(len(np.unique(self.segments)), dtype=int)
        # Setting the mask entries corresponding to the picked indices to zero
        decisions[highest_indices] = 1  # Create perturbed image using the mask
        mask = np.isin(self.segments, unique_segments[decisions == 1])

        perturbed_image = self.image * torch.tensor(mask, dtype=torch.float32)
        output = self.model(self.image)
        output = self.get_top_predictions(output)
        class_imagenet = output[prediction_index]
        class_name = self.class_names[class_imagenet]
        #output_class = dataset.classes[output]
        self.plot_image(perturbed_image, class_name)
    def get_top_predictions(self, prediction, top_n=2):
        """
        Gets the top N predictions from the model output.

        Args:
            prediction (torch.Tensor): Model prediction output.
            top_n (int): Number of top predictions to return.

        Returns:
            list: List of top N class indices.
        """
        probs = torch.nn.functional.softmax(prediction, dim=1)
        top_probs, top_classes = torch.topk(probs, top_n)
        return top_classes.squeeze().tolist()


def exponential_kernel(similarity, width):
    return np.exp(-(similarity ** 2) / (2 * (width ** 2)))

if __name__ == "__main__":
    # Load the pre-trained model using the new `weights` parameter

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
    model.eval()

    # Assume model is a predefined model and image_path the location of the image
    lime_instance = Lime(model)
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    for im in images:
        lime_instance.load_image(im)
        lime_instance.generate_superpixels()
        lime_instance.create_perturbed_images_and_predict(num_pertubations=200)
        coeffs = lime_instance.fit_regression()
        lime_instance.visualize_impact(prediction_index=1)
        lime_instance.delete_memory()

