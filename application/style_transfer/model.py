import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

import random
import _thread
import time
import flask

import requests

# UTILS
from .image_util import load_img, tensor_to_image


class Settings:
    """
    used to set settings for model
    """
    BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
    CONTENT_FOLDER = BASE_DIR + "/static/images/content/"
    STYLE_FOLDER = BASE_DIR + "/static/images/style/"
    OUTPUT_FOLDER = BASE_DIR + "/static/images/output/"

    def __init__(self):
        self.content_image_path = None
        self.style_image_path = None
        self.content_image = None  # image_util.load_image()
        self.style_image = None  # image_util.load_image()
        self.variation = 30
        self.style_weight = 1e-2
        self.content_weight = 1e4
        self.learning_rate = 0.02
        self.beta = 0.99
        self.epsilon = 1e-1

        # Load the VGG19 model so we can use its first few layers that are already trained
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        # Now pick the layers we want to use from the VGG model

        # Content layer where will pull our feature maps
        self.content_layers = ['block5_conv2'] 

        # Style layer of interest
        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1',]


    def set_content_image(self, file_name):
        """
        Sets the content file name attribute and loads the file

        :param file_name: string
        :return: bool
        """
        self.content_image_path = file_name
        try:
            self.content_image = load_img(self.CONTENT_FOLDER + file_name)
            return True
        except:
            return False

    def set_style_image(self, file_name):
        """
        Sets the style file name attribute and loads the file

        :param file_name: string
        :return: bool
        """
        self.style_image_path = file_name
        try:
            self.style_image = load_img(self.STYLE_FOLDER + file_name)
            return True
        except:
            return False

    def num_content_layers(self):
        """
        returns number of content layers
        """
        return len(self.content_layers)

    def num_style_layers(self):
        """
        returns number of style layers
        """
        return len(self.style_layers)

    def set_content_layers(self, layers):
        """
        set the layers that will be used from vgg for content image

        :param layers: list

        Raises: invalid argument error
        """
        # TODO check if layers are valid
        self.content_layers = layers
       
    def set_style_layers(self, layers):
        """
        set the layers that will be used from vgg to style image

        :param layers: list

        Raises: invalid argument error
        """
        # TODO check if layers are valid
        self.style_layers = layers

    def set_style_weight(self, value):
        self.style_weight = value

    def set_content_weight(self, value):
        self.content_weight = value

    def set_variation(self, value):
        """
        set the total training variation
        """
        if value <= 0:
            raise Exception("Value must be above 0")

        self.variation = value

    def __repr__(self):
        return f"Settings(Content Image={self.content_image_path} Style Image={self.style_image_path} Variance={self.variation} Content Weight={self.content_weight} Style Weight={self.style_weight})"


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = StyleContentModel.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    @staticmethod
    def gram_matrix(input_tensor):
        """
        calculate the style of an input tensor using a gram matrix

        :param input_tensor: tf.Tensor
        :return: float
        """
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    @staticmethod
    def vgg_layers(layer_names):
        """ 
        Creates a vgg model that returns a list of intermediate output values.

        :param layer_names: list of str
        :return: keras.model
        """

        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)  # ask the model to return intermediate outputs at the layers we passed in
        return model

    def call(self, inputs):
        "Expects float input in [0,1]"

        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [StyleContentModel.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}


class Model:
    def __init__(self, settings):
        """
        :param settings: Settings object
        """
        self.settings = settings

        self.content_weight = settings.content_weight
        self.style_weight = settings.style_weight
        self.total_variation_weight = self.settings.variation

        self.num_style_layers = settings.num_style_layers()
        self.num_content_layers = settings.num_content_layers()

        self.image = tf.Variable(settings.content_image)

        lr = settings.learning_rate
        beta = settings.beta
        epsilon = settings.epsilon
        self.opt = tf.optimizers.Adam(learning_rate=lr, beta_1=beta, epsilon=epsilon)

        self.extractor = StyleContentModel(settings.style_layers, settings.content_layers)

        self.style_targets = self.extractor(settings.style_image)['style']
        self.content_targets = self.extractor(settings.content_image)['content']


    @staticmethod
    def high_pass_x_y(image):
        x_var = image[:,:,1:,:] - image[:,:,:-1,:]
        y_var = image[:,1:,:,:] - image[:,:-1,:,:]

        return x_var, y_var

    @staticmethod
    def total_variation_loss(image):
        x_deltas, y_deltas = Model.high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    @staticmethod
    def clip_0_1(image):
        """
        Keep pixel values between 0 and 1

        :param image: array
        :return: array
        """
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    @tf.function()
    def train_step(self, image):
        """
        runs one step of the training process on the image

        :param image: tf.Tensor
        """
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(Model.clip_0_1(image))

    def style_content_loss(self, outputs):
        """
        calculates loss between predicted style and new style
        """

        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss

        return loss

    def generate(self):
        """
        generate new image

        :param steps: number of steps to run for
        :returns: PIL image
        """
        
        self.train_step(self.image)
        return tensor_to_image(self.image)

    def save_file(self):
        """
        saves the file as a combination of the style and content image
        name into the foulder specified in output.
        """
        file_name = self.settings.OUTPUT_FOLDER + self.settings.content_image_path + "-" +self.settings.style_image_path + str(random.randint(0,10000)) + ".png"
        tensor_to_image(self.image).save(file_name)

