# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.                                                                                  
#                                                                                                                                              
# Licensed under the Apache License, Version 2.0 (the "License");                                                                              
# you may not use this file except in compliance with the License.                                                                             
# You may obtain a copy of the License at                                                                                                      
#                                                                                                                                              
#     http://www.apache.org/licenses/LICENSE-2.0                                                                                               
#                                                                                                                                              
# Unless required by applicable law or agreed to in writing, software                                                                          
# distributed under the License is distributed on an "AS IS" BASIS,                                                                            
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                                                                     
# See the License for the specific language governing permissions and                                                                          
# limitations under the License.                                                                                                               
# ==============================================================================                                                               

import tensorflow as tf
import horovod.tensorflow.keras as hvd


import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from datetime import datetime


# Horovod: initialize Horovod.                                                                                                                 
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)                                                                      
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    #tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')                                                                
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()%len(gpus)], 'GPU')

array_file = open('array.npy', 'rb')
indata = np.load(array_file)
array_file.close()

stride_data = int(150000/hvd.size())
x_train = indata[hvd.rank()*stride_data:(hvd.rank()+1)*stride_data]
x_test  = indata[150000:]

del indata

input_img = tf.keras.Input(shape=(x_train.shape[-1],))
encoded = tf.keras.layers.Dense(512, activation='relu')(input_img)
encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
#encoded = layers.Dense(8, activation='relu')(encoded)                                                                                         

decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
#decoded = layers.Dense(64, activation='relu')(decoded)                                                                                        
decoded = tf.keras.layers.Dense(x_train.shape[-1], activation='relu')(decoded)

loss_fn = tf.keras.losses.MeanSquaredError()

# Horovod: adjust learning rate based on number of GPUs.                                                                                       
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.                                                                                                   
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow                                                                   
# uses hvd.DistributedOptimizer() to compute gradients.                                                                                        
#mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),                                                                           
#                    optimizer=opt,                                                                                                            
#                    metrics=['accuracy'],                                                                                                     
#                    experimental_run_tf_function=False)                                                                                       

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer=opt, loss=loss_fn)
#autoencoder.compile(loss=tf.losses.SparseCategoricalCrossentropy(),           #                                                               
#                    optimizer=opt,                                            #                                                               
#                    metrics=['accuracy'],                                     #                                                               
#                    experimental_run_tf_function=False)                                                                                       

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.                                                           
    # This is necessary to ensure consistent initialization of all workers when                                                                
    # training is started with random weights or restored from a checkpoint.                                                                   
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.                                                                        
    #                                                                                                                                          
    # Note: This callback must be in the list before the ReduceLROnPlateau,                                                                    
    # TensorBoard or other metrics-based callbacks.                                                                                            
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final                                                      
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during                                                         
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.                                                                
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.                                                    

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.                                                    

#if hvd.rank() == 0:                                                                                                                           
#    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))                                                           

# Horovod: write logs on worker 0.                                                                                                             
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.                                                                                                                             
# Horovod: adjust number of steps based on number of GPUs.                                                                                     
#mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)


start_time = datetime.now()

autoencoder.fit(x_train, x_train, steps_per_epoch=500 // hvd.size(),
                callbacks=callbacks, epochs=6, validation_steps=10, validation_data=(x_test, x_test),verbose=verbose)

# do your work here

end_time = datetime.now()
print('Duration[{}]: {} train_len {} test_len {}'.format(hvd.rank(),end_time - start_time,len(x_train),len(x_test)))

