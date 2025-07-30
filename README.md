Face Recognition for PTB (Telegram) using OpenCV

The program uses a trained Caffe model Arcface.onnx to detect faces. For detection of known faces, photos of people you want the program to detect are placed inside folders. The program then compares the faces inside the folder and faces on frames of incoming video, then returns thr name if a threshold is met.




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
        
