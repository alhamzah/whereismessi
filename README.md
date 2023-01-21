# whereismessi

SDAIA Smartathon Challenge 2023

# Pothole Detector Demo

You can see our pothole detector in action [here](https://drive.google.com/file/d/1JNgdzlMls3zSHuT-1xMyhmUg-zViEA3t/view?usp=sharing) 

# Datasets

The datasets used to train the models can be found [here](https://drive.google.com/drive/folders/11L_LIEgdhYwhXSZ2Z62EUtn6oF644ras?usp=sharing)

# Model weights

TODO

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
