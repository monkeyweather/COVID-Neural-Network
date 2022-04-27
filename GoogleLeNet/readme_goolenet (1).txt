Project method:       GoogLeNet (Inception v3)


Group member:        Wenhui Liu, Yibo Li, Manita Ngarmpaiboonsombat
Group member email:  Wenhui_Liu@student.uml.edu    Yibo_Li@student.uml.edu
		     Manita_Ngarmpaiboonsombat@student.uml.edu


Description: Before building GoogLeNet, we used Contrast Limited Adaptive Histogram Equalization
	     on all images. This makes the chest X-ray image clearer, which can help us better 
  	     catch the features of data. Then, we decide to use Inception v3 pre-trained model 
	     from keras library and t-learning the model based on our dataset. Then we only train
  	     the last 20 layers of module (considering t-learning first, then apply fine -tuning 
	     with a small learning-rate). Next, we use GlobalAveragePooling2D to get rid of noise
	     and redundant features. Finally, we use the fully connected layer (Dense), to extract
	     the features correlation, map to the output, and use the softmax activation function 
	     in it. After this, we also set the Adam optimizer and Loss function. Then get the 
	     accuracy, precision, recall and f1-score through the confusion matrix, which can help 
	     us better analyze the performance of our model.
 	     In addition, we actually ran 3 different pre-trained models from Keras library: Inception
	     v3, Inception V3 + ResNet, and VGG-19, we also tried transfer learning and fine tuning on 
	     Inception v3 model.




How to run: This is a .ipynb file, and you can see the results of each step when you open it.
            If you want to rerun, run each cell from the beginning to the end.