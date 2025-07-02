PACAKGES AND VERSIONS :
tensorflow 2.8.1
opencv 3.3 +
mediapipe (natural)
matplotlib (natural)
scikit learn (natural)

!pip install tensorflow==2.8.1 tensorflow-gpu==2.8.1 opencv-python mediapipe scikit-learn matplotlib


WORKING :
Model Construction :
( functions (def ) are written for some part of training such as calling in/collecting key-points using media pipe so it can be used in all calls in this software )
( further all the data collected will be stored in Numpy arrays )

Model Training : Deep Neural Network with LSTM layers ( after this we can predict a sign using number frame range  )

Model Testing : Real time sign language prediction using our webcam 

Model Evaluation : Based on not only poses but all so the complete pose cycle which means every pose will have a build up to drop cycle (action) 






