**Things We Need to Do**

1. Setup APP Folder
    * Create a folder named 'app' into your root directory
2. Install Kivy
    * Run: pip install kivy[full] kivy_examaples
    * Check installation with: pip list
3. Setup Validation Folder
    * Copy the application_data folder and paste into app folder
4. Create Custon Layer Module
    * Import TensorFlow and Layer
    * Copy the L1Dist class created on the notebook
5. Bring over h5 Moldel
    * Copy the .h5 file creted at the root folder and paste it into the app folder

6. Create faceId.py file
    * Import Dependencies for Kivy
        - Kivy Dependencies: App, BoxLayout, Image, Button, Label, Clock, Texture, Logger
        - Other Dependencies: cv2, TensorFlow, L1Dist, os, numpy
    * Build Layout
    * Build Update Function
    * Bring over Preprocessing Function

10. Bring over Verification Function
11. Update Verification Function to handle new paths and save current frame
12. Update Verification Function to set verified text
13. Link Verification Function to Button
14. Setup Logger 
