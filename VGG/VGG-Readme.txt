Project method: VGG16

Group members: Shane Corson, Shane Calla
Group member emails: shane_corson@student.uml.edu shane_calla@student.uml.edu

Description: We follow a similar data preprocessing strategy from GoogleLeNet, except we do not use CLAHE.
We then use Keras's VGG to set up the base model of the architecture we will be using. We add a few more
layers including a 2D average pooling layer, a flattening layer, a densely connected layer, dropout, and 
another densely connected layer. We then print out a confusion matrix. The vgg uses the Adam optimizer with
categorical crossentropy loss and a learning rate of .0001. Other methods were attempted including different
layers and learning rates, however these were the best results, and was a common setup.

Instructions: All of the imports are at the top block of runnable code. Then, there is a set of code to see
if CoLab is connected to the GPU. The next section is meant to mount the drive to the gpu so we can 
connect the data directly from the Kaggle. The data download section only needs to be run once, then it can be
ignored. Setting Data Paths is ran followed by Preprocessing, and then the VGG is actually set up in the next
section. Each of the sections can be ran from here.

Things to note: We were planning on doing more epochs, however google CoLab did not let us use the GPU's
despite us paying for CoLab pro. Also, the confusion matrix code is based on the code from the other group in
our group, as well as the preprocessing data. The working code we showed was just to show that it works,
we also have a seperate output from the 10 epoch run.

Much was inspired from https://www.kaggle.com/code/phannghia2306/covid-19-vgg16
https://www.kaggle.com/code/rohitadnaik/transferlearning-vgg16-xray-classification
https://www.kaggle.com/code/kartik2khandelwal/covid-pneumonia-detector-using-vgg16/notebook


