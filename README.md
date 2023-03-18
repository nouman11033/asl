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


REMARKS :
#andrew ing was a god in this ,64 128 neurons

my lstm system : 

model = Sequential() #api, just adding a bunch of layers for each 
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #true cuz next layer will use it ,
#with a activation component and a basic shape 30 frames x 1662 values which is x .shape last two array points
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) #return activation will be 0 to 1 with sum of all values adding upto 1


#models right now use no of cnn layers/pre tained mobile net with lstm layers ,30 x3 class 90 sequences will give no level of accuracy
#cuz of numaan being lazy and his less data with just 3 sequences and we have to create a hyper accurate model
#much denser neural network 30-50 million parameters go right up to 1.5 million parameters now 
#now 1.5 million netowrk was very much simplar so it was fast
#also back then i was working on my mac and it worked well okay but due to introduction of media pipe
#my venture started lossing its pipeline value as media pipe has its own open cv which is sdk loaded which doesn't work w mac
#also tensorflow c++ desktop tools via community vs code is not applicable for mac


