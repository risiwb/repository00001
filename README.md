Face Recognition for PTB (Telegram) using OpenCV


for a proper work arcface.onnx and deploy.prototxt should be downloaded and placed in the same folder with your code.

photos of faces should be saved inside a folder "{Name} {Surname}" inside a folder "known_faces" inside a folder with your code.
Example: 
- PROJECT (folder)
- - bot.py
  - arcface.onnx
  - deploy.prototxt
  - known_faces (folder)
  - - John Smith (folder)
      - Photo
      - Photo
  - - Bob Red (folder)
      - Photo
        
