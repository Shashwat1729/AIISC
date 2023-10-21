#Dependencies
#--- Before using the below code install CLIP : pip install git+https://github.com/openai/CLIP.git
import clip
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')
# tf.keras.utils.disable_interactive_logging()# Disable logging for Tensorflow

class ImageSimilarity:
    def __init__(self):
        # Initialize CLIP model and VGG model here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.vgg_model = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))
        for model_layer in self.vgg_model.layers:
            model_layer.trainable = False

    def load_image(self, img):
        resized_image = img.resize((224, 224))
        return resized_image

    def get_image_embeddings_vgg(self, object_image):
        image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
        image_embedding = self.vgg_model.predict(image_array)
        return image_embedding

    def get_image_embeddings_clip(self, image_1, image_2):
        image_1 = self.clip_preprocess(image_1).unsqueeze(0).to(self.device)
        image_2 = self.clip_preprocess(image_2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features_1 = self.clip_model.encode_image(image_1).cpu().detach().numpy()
            image_features_2 = self.clip_model.encode_image(image_2).cpu().detach().numpy()
            cosine_similarity_score = cosine_similarity(image_features_1, image_features_2).reshape(1,)
        return cosine_similarity_score[0]

    def get_similarity_score(self, image_1, image_2, model_name="VGG"):
        if model_name == "CLIP":

            return self.get_image_embeddings_clip(image_1, image_2)
        else:
            image_1 = self.load_image(image_1)
            image_2 = self.load_image(image_2)
    

            image_feature_1 = self.get_image_embeddings_vgg(image_1)
            image_feature_2 = self.get_image_embeddings_vgg(image_2)
            similarity_score = cosine_similarity(image_feature_1,image_feature_2).reshape(1,)

            return similarity_score[0]
