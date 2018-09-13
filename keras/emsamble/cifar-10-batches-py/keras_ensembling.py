
# coding: utf-8

# # Ensembling ConvNets using Keras
# ## Introduction
# "In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Unlike a statistical ensemble in statistical mechanics, which is usually infinite, a machine learning ensemble consists of only a concrete finite set of alternative models, but typically allows for much more flexible structure to exist among those alternatives." \[[Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning)\]
# 
# The main motivation for using an ensemble is to find a hypothesis that is not necessarily contained within the hypothesis space of the models from which it is built. Empirically, ensembles tend to yield better results when there is a significant diversity among the models.
# 
# ## Motivation
# 
# If you look at results of any big machine learning competition, you will most likely find that the top results are achieved by an ensemble of models rather than a single model. For instance, the top-scoring single model architecture at [ILSVRC2015](http://www.image-net.org/challenges/LSVRC/2015/results) is on place 13. Places 1-12 are taken by various ensembles.
# 
# I haven't seen a tutorial or documentation on how to use several neural networks in an ensemble, so I decided to share the way I do it. I will be using Keras, specifically its Functional API, to recreate three small CNNs (compared to ResNet50, Inception etc.) from relatively well-known papers. I will train each model separately on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) training dataset. Then each model will be evaluated using the test set. After that, I will put all three models in an ensemble and evaluate it. It is expected that the ensemble will perform better on a test set that any single model in the ensemble separately.
# 
# There are many different types of ensembles; stacking is one of them. It is one of the more general types and can theoretically represent any other ensemble technique. Stacking involves training a learning algorithm to combine the predictions of several other learning algorithms. For the sake of this example, I will use one of the simplest forms of Stacking, which involves taking an average of outputs of models in the ensemble. Since averaging doesn't take any parameters, there is no need to train this ensemble (only its models).

# ## Preparing the data
# First, import dependencies.

# In[1]:


from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10

import numpy as np


# I am using CIFAR-10, since it is relatively easy to find papers describing architectures that work well on this dataset. Using a popular dataset also makes this example easily reproducible.
# 
# Here the dataset is imported. Both train and test image data is normalized. The training label vector is converted to a one-hot-matrix. Don't need to convert the test label vector, since it won't be used during training.

# In[8]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)


# In[7]:


train_size=100
test_size=int(train_size/10)
x_train=x_train[0:train_size]
y_train=y_train[0:train_size]

x_test=x_test[0:test_size]
y_test=y_test[0:test_size]


# The dataset consists of 60000 32x32 RGB images from 10 classes. 50000 images are used for training/validation and the other 10000 for testing.

# In[9]:


print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,
                                                                                          x_test.shape, y_test.shape))


# Since all three models work with the data of the same shape, it makes sense to define a single input layer that will be used by every model.

# In[10]:


input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)


# ## First model: ConvPool-CNN-C
# 
# The first model that I am going to train is ConvPool-CNN-C \[[Springenberg et al., 2015, Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)\]. Its descritption appears on page 4 of the linked paper.
# 
# The model is pretty straightforward. It features a common pattern where several convolutional layers are followed by a pooling layer. The only thing about this model that might be unfamiliar to some people is its final layers. Instead of using several fully-connected layers, a global average pooling layer is used. 
# 
# Here is a brief overview of how global pooling layer works. The last convolutional layer `Conv2D(10, (1, 1))` outputs 10 feature maps corresponding to ten output classes. Then the `GlobalAveragePooling2D()` layer computes spatial average of these 10 feature maps, which means that its output is just a vector with a lenght 10. After that, a softmax activation is applied to that vector. As you can see, this method is in some way analogous to using FC layers at the top of the model. You can read more about global pooling layers and their advantages in [Network in Network](https://arxiv.org/abs/1312.4400) paper.
# 
# One important thing to note: there's no activation function applied to the output of the final `Conv2D(10, (1, 1))` layer, since the output of this layer has to go through `GlobalAveragePooling2D()` first.

# In[11]:


def conv_pool_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model


# In[12]:


conv_pool_cnn_model = conv_pool_cnn(model_input)


# For simplicity's sake, each model is compiled and trained using the same parameters. Using 20 epochs with a batch size of 32 (1250 steps per epoch) seems sufficient for any of the three models to get to some local minima. Randomly chosen 20% of the training dataset is used for validation.

# In[13]:


def compile_and_train(model, num_epochs): 
    
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32, 
                     epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history


# It takes about 1 min to train this and the next model for one epoch using a single Tesla K80 GPU. Training might take a while if you are using a CPU.

# In[ ]:


_ = compile_and_train(conv_pool_cnn_model, num_epochs=20)


# One simple way to evaluate the model is to calculate the error rate on the test set.

# In[9]:


def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]    
    return error


# In[22]:


evaluate_error(conv_pool_cnn_model)


# ## Second model: ALL-CNN-C
# 
# The next CNN, ALL-CNN-C, comes from the same paper \[[Springenberg et al., 2015, Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)\]. This model is very similar to the previous one. Really, the only difference is that convolutional layers with a stride of 2 are used in place of max pooling layers. Again, note that there is no activation function used immediately after the `Conv2D(10, (1, 1))` layer. The model will fail to train if a `relu` activation is used immediately after that layer.

# In[10]:


def all_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model


# In[11]:


all_cnn_model = all_cnn(model_input)


# In[ ]:


_ = compile_and_train(all_cnn_model, num_epochs=20)


# Since two models are very similar to each other, it is expected that the error rate doesn't differ much.

# In[23]:


evaluate_error(all_cnn_model)


# ## Third Model: Network In Network CNN
# 
# The third CNN is Network in Network CNN \[[Lin et al., 2013, Network In Network](https://arxiv.org/abs/1312.4400)\]. This is a CNN from the paper that introduced global pooling layers. It's smaller than previous two models, therefore is much faster to train. No `relu` after the final convolutional layer!

# In[12]:


def nin_cnn(model_input):
    
    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    
    return model


# In[13]:


nin_cnn_model = nin_cnn(model_input)


# This model trains much faster -  15 seconds per epoch on my machine.

# In[ ]:


_ = compile_and_train(nin_cnn_model, num_epochs=20)


# This is more simple than the other two, so the error rate is a bit higher.

# In[24]:


evaluate_error(nin_cnn_model)


# ## Three Model Ensemble
# 
# Now all three models will be combined in an ensemble. 
# 
# Here, all three models are reinstantiated and the best saved weights are loaded.

# In[14]:


conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

import glob
import os


# conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.29-0.10.hdf5')
# all_cnn_model.load_weights('weights/all_cnn.30-0.08.hdf5')
# nin_cnn_model.load_weights('weights/nin_cnn.30-0.93.hdf5')

conv_pool_cnn_model.load_weights(max(glob.glob('weights/conv_pool_cnn*'), key=os.path.getctime))
all_cnn_model.load_weights(max(glob.glob('weights/all_cnn*'), key=os.path.getctime))
nin_cnn_model.load_weights(max(glob.glob('weights/nin_cnn*'), key=os.path.getctime))

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]


# Ensemble model definition is very straightforward. It uses the same input layer thas is shared between all previous models. In the top layer, the ensemble computes the average of three models' outputs by using `Average()` merge layer.

# In[15]:


def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model


# In[16]:


ensemble_model = ensemble(models, model_input)


# As expected, the ensemble has a lower error rate than any single model.

# In[17]:


evaluate_error(ensemble_model)


# ## Other Possible Ensembles
# 
# Just for completeness, we can check performance of ensembles that consist of 2 model combinations.

# In[18]:


pair_A = [conv_pool_cnn_model, all_cnn_model]
pair_B = [conv_pool_cnn_model, nin_cnn_model]
pair_C = [all_cnn_model, nin_cnn_model]


# In[19]:


pair_A_ensemble_model = ensemble(pair_A, model_input)
evaluate_error(pair_A_ensemble_model)


# In[20]:


pair_B_ensemble_model = ensemble(pair_B, model_input)
evaluate_error(pair_B_ensemble_model)


# In[21]:


pair_C_ensemble_model = ensemble(pair_C, model_input)
evaluate_error(pair_C_ensemble_model)


# ## Conclusion
# 
# To reiterate what was said in the introduction: every model has its own weaknesses. The reasoning behind using an ensemble is that by stacking different models representing different hypotheses about the data, we can find a better hypothesis that is not in the hypothesis space of the models from which the ensemble is built.
# 
# By using a very basic ensemble, a much lower error rate was achieved than when a single model was used. This proves effectiveness of ensembling.
# 
# Of course, there are some practical considerations to keep in mind when using an ensemble for your machine learning task. Since ensembling means stacking multiple models together, it also means that the input data needs to be forward-propagated for each model. This increases the amount of compute that needs to be performed and, consequently, evaluation (predicition) time. Increased evaluation time is not critical if you use an ensemble in research or in a Kaggle competition. However, it is a very critical factor when designing a commercial product. Another consideration is increased size of the final model which, again, might be a limiting factor for ensemble use in a commercial product.
