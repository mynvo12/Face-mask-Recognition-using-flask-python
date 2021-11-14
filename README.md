# Face-Mask-Recognition-web-app-with-Flask
 
+ This project is based on https://github.com/chandrikadeb7/Face-Mask-Detection , but with 3 classes(with_mask,without_mask,incorrect_mask)


+ Project is using Tensorflow, keras to train model with format HDF5, and use flask framework to show the real time Face Mask Recognition 

+ This project using Python 3.8.10 and GPU: GTX-1650TI with CUDA support: v11.2 and CUDNN: v8.1.1



![image](https://user-images.githubusercontent.com/74602408/141684105-ead6da98-6888-4fcf-849b-006f3cc334fa.png)

+ So firstly, you have to download the project and then install requirement.txt 
 following this command line: pip install -r requirement.txt
 
+ So the code training and demo in folder source: 
Run "python Training.py" to train your own model with your own dataset
Run "python demo.py" to inference your result or my result model with webcam 

+ The code for web inference in the folder: Flask
Run "python app.py" to inference in your local host

My model have a good accurracy: 0.98 with 3 classes!!!!!

So good luck and enjoy the result 

![image](https://user-images.githubusercontent.com/74602408/141684297-275b4fa3-33b4-4d7e-a2d5-af3e0ae3ea5e.png)



