# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.logger import Logger
from kivy.lang import Builder

# Import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
from layers import L1Dist

Builder.load_file('label_color.kv')

class RoundedButton(Button):
    pass

class MyLogo(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 1, 0, 0.25)
            Rectangle(size=self.size)

class BlackLabel(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0, 0, 0.25)
            Rectangle(size=self.size)

# Build app and layout
class CamApp(App):

    def build(self):

        blue = [0.1, 0, 1, 1]

        # Main layout components
        self.logo = MyLogo(text="Face Recognizer APP", size_hint=(1,.15))
        self.web_cam = Image(size_hint=(1,.8))
        self.button = RoundedButton(text="Click to Verify", on_press=self.verify, background_color=blue, size_hint=(0.5, .15), pos_hint={'center_x': 0.5})
        self.black_label = BlackLabel(text="", size_hint=(1,.05))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .15))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.logo)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.black_label)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamese_model_400_30.h5', custom_objects={'L1Dist':L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        Logger.info("Capturing...")
        
        return layout
    
    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def pre_process(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.7
        verification_threshold = 0.7

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.pre_process(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.pre_process(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) >= detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification >= verification_threshold

        # Set verification text and color
        red = [1, 0, 0, 1]
        green = [0, 1, 0, 1]
        # self.verification_label.text = 'Verified! Access granted!' if verified == True else 'Unverified! Access Denied!'
        self.verification_label.text = 'Verified!' if verified == True else 'Unverified!'
        self.verification_label.background_color = green if verified == True else red

        # Log out details
        Logger.info(results)
        Logger.info('Total images: {0}'.format(len(results)))
        Logger.info('Nº similar images: {0}'.format(detection))
        Logger.info('Verification value: {0}'.format(verification))
        Logger.info('Verified: {0}'.format(verified))

        
        return results, verified

if __name__ == '__main__':
    CamApp().run()
