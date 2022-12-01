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
# from kivy.graphics.vertex_instructions import RoundedRectangle
from kivy.graphics.vertex_instructions import RoundedRectangle
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
    # def on_size(self, *args):
    #     self.canvas.before.clear()
    #     with self.canvas.before:
    #         Color(self.color)
    #         RoundedRectangle(size=self.size, pos=self.pos, radius=[50])
    pass

class MyLogo(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0, 0, 0)
            Rectangle(size=self.size)

class ResultLabel(Label):
    pass

class MainLayout(BoxLayout):
    pass

# Build app and layout
class CamApp(App):

    def build(self):
        
        self.total_images = len(os.listdir(os.path.join('application_data', 'verification_images')))

        # Colors to use
        self.color_gray = [134/255, 135/255, 134/255, 1]
        self.color_cream = [184/255, 104/255, 104/255, 1]
        self.color_cream2 = [179/255, 93/255, 93/255, 1]
        self.color_cian = [6/255, 168/255, 204/255, 1]
        self.color_blue = [19/255, 141/255, 168/255, 1]
        self.red = [191/255, 25/255, 42/255, 1]
        self.green = [0, 1, 0, 1]

        # Main layout components
        self.margin_top = Label(text="", size_hint=(1, .1))
        self.logo = MyLogo(text="Face Recognizer APP", size_hint=(1,.15))
        self.web_cam = Image(size_hint=(1,.8))
        self.button = RoundedButton(text="Verify!",  on_press=self.trigger_verify, size_hint=(0.10, .20), pos_hint={'center_x': 0.5})
        self.middle_label = Label(text="", size_hint=(1,.05))
        self.verification_label = ResultLabel(text="Verification unitialized", background_color=self.color_cream2, size_hint=(0.40, .125), pos_hint={'center_x': 0.5})
        self.margin_botton = Label(text="", color=(1, 1, 1, 1), size_hint=(1, .15))

        # Add items to layout
        layout = MainLayout(orientation='vertical')
        layout.add_widget(self.margin_top)
        layout.add_widget(self.logo)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.middle_label)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.margin_botton)

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

    def trigger_verify(self, *args):
        self.button.text = 'Wait'
        self.verification_label.text = 'Verifying...'
        Clock.schedule_once(self.verify, 0.5)

    def trigger_reset(self, p=''):
        Clock.usleep(1500000)
        self.button.text = 'Verify!'
        self.verification_label.text = 'Verification unitialized' 
        self.verification_label.background_color = self.color_cream2

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
        self.count_images = 0
        verification_images = os.listdir(os.path.join('application_data', 'verification_images'))
        for image in verification_images:
            self.count_images += 1
            Logger.info('Processing {0}/{1} image'.format(self.count_images, len(verification_images)))
                
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

        self.verification_label.text = 'Verified!' if verified == True else 'Unverified!'
        self.verification_label.background_color = self.green if verified == True else self.red

        # Log out details
        Logger.info('Total images: {0}'.format(len(results)))
        Logger.info('Nº similar images: {0}'.format(detection))
        Logger.info('Verification value: {0}'.format(verification))
        Logger.info('Verified: {0}'.format(verified))

        Clock.schedule_once(self.trigger_reset, 5)
        
        return results, verified

if __name__ == '__main__':
    CamApp().run()