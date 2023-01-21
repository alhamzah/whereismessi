# whereismessi

SDAIA Smartathon Challenge 2023

# Pothole Detector Demo

You can see our pothole detector in action [here](https://www.youtube.com/embed/hjwStrxPk9Y) 

# Datasets

The datasets used to train the models can be found [here](https://drive.google.com/drive/folders/11L_LIEgdhYwhXSZ2Z62EUtn6oF644ras?usp=sharing)

# Model weights

The model weights can be found [here](https://drive.google.com/drive/folders/1fMcpEK5xWA-J3CkYtQLVfOQTyV1WTP2O?usp=sharing) 

# Inference

To run inference using the PCI regressor:
```
model = keras.models.load_model(<path/to/model>)
model.predict(<images>)
```

To run inference using the pothole classifier or the pothole detector:
```
python yolov7/detect.py --source <path_to_video> --weights <path_to_weights> --view-img
```
