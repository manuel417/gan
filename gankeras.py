# import section

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers

import time

# GAN Generator - a CNN that generates images from a vector of random noise. Vector has 100 dimension
def make_generator_model():
    model = tf.keras.Sequential()
    # input layer - vector with 100 dimension (random noise)
    # Dense layer with 7x7x256 neurons
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape = (100,)))
    # Normalize since we do not have activation in previous layer
    model.add(layers.BatchNormalization())
    # Add activation - Leaky Relu
    model.add(layers.LeakyReLU())

    # Reshape this thing to (None, 7 , 7, 256) which means: batch of unspecified size (i.e., None) with
    # images of size 7 x 7 x 256 (i.e., 256 channels). Upcoming layers will grow the image and reduce
    # number of channels. NOTE: in general in TF and in Keras the first dimension of the data set
    # is the number of samples, so they use while building the NN None because the number of samples
    # is not known in advance.
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7 , 7 , 256)

    # Add convolutional transpose layer with 128 5x5 filters, a 1x1 stride, and same padding.
    # Convolutional transpose is used for upsampling images from lower to higher resolution.
    model.add(layers.Conv2DTranspose(128, (5,5), strides = (1,1), padding='same', use_bias=False)) #why not use Bias???
    assert model.output_shape == (None, 7 , 7 , 128)
    # Normalize since we do not have activation in previous layer
    model.add(layers.BatchNormalization())
    # Add activation - Leaky Relu
    model.add(layers.LeakyReLU())

    # Add convolutional transpose layer with 64 5x5 filters, a 2x2 stride, and same padding.
    model.add(layers.Conv2DTranspose(64, (5,5), strides =  (2,2), padding='same', use_bias=False)) #why not use Bias???
    assert model.output_shape == (None, 14 , 14 , 64)
    # Normalize since we do not have activation in previous layer
    model.add(layers.BatchNormalization())
    # Add activation - Leaky Relu
    model.add(layers.LeakyReLU())

    # Add Final convolutional transpose layer with 1 5x5 filters, a 2x2 stride, and same padding.
    # This will produce the 28 x 28 x 1 image
    model.add(layers.Conv2DTranspose(1, (5,5), strides =  (2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28 , 28 , 1)

    return model

## Discriminator network

def make_discriminator_model():
    model = tf.keras.Sequential()

    # Convlutional network that has two conv layers to classify the image as real or fake.

    # Convolutinal layer with 64 5x5 filter, stride of 2 and same padding.
    # input is the 28 x 28 x 1 image
    model.add(layers.Conv2D(64, (5,5), strides = (2,2), padding='same', input_shape=[28, 28, 1]))
    # Leaky ReLU actvation
    model.add(layers.LeakyReLU())
    # Dropuout - regularization layer to prevent NN to memorize images
    # This will drop 30% of neurons output
    model.add(layers.Dropout(0.3))

    # Convolutinal layer with 128 5x5 filter, stride of 2 and same padding.
    # input is the 28 x 28 x 1 image
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten layer
    model.add(layers.Flatten())

    # final layer with 1 neuron
    model.add(layers.Dense(1, activation='sigmoid'))  # why this does not have an activation??

    return model

## convenience function to compute the cross_entropy
def get_cross_entropy():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Loss function for the discriminator
def discriminator_loss(real_output_predictions, fake_output_predictions):
    # compute cross entropy between real images and predicted
    cross_entropy = get_cross_entropy()

    # vector with all 1s and same shape as predicted output on real images - this is what the discriminator
    # should have predicted on real images, all 1s
    real_result = tf.ones_like(real_output_predictions)
    # error for real images
    real_loss = cross_entropy(real_result, real_output_predictions)

    # vector with all 0s and same shape as predicted output on fake images- this is what the discriminator
    # should have predicted on fake images, all 0s
    fake_result = tf.zeros_like(fake_output_predictions)
    # error for fake images
    fake_loss = cross_entropy(fake_result, fake_output_predictions)

    #total loss = sum of all losses
    total_loss = real_loss + fake_loss

    return total_loss

# Loss function for the generator

def generator_loss(fake_output_predictions):
    # computer cross entropy on predicted images
    cross_entropy = get_cross_entropy()

    # vector with all 1s and same shape as predicted output on fake images - this is what the generator
    # wants the discriminator to think, that all  fake images are real, and are all 1s
    fake_result = tf.ones_like(fake_output_predictions)
    # error for fake images
    fake_loss = cross_entropy(fake_result, fake_output_predictions)
    return fake_loss

# This function defines a training step in the training process.
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator, generator_optimizer, discriminator, discriminator_optimizer, real_images, batch_size, noise_dim):
    # Noise input tensor: it is a matrix . batch size
    # is the number of rows, which are de examples de generate
    # and noise dim is the dimension of the noise vector.
    noise = tf.random.normal([batch_size, noise_dim])

    # Gradient Tape is an object used to collect and update gradients
    # in the NN. It is one of those magic singletons ...
    # Here we need one tape per NN - gen_tape is for the generator
    # and disc_tape is for the discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate a batch of fake images
        # Recall: generator starts with a noise vector and after training enough times ends up
        # using it as a starting point to draw a fake image that looks real
        generated_images = generator(noise, training=True)

        # detect real images
        real_output = discriminator(real_images, training=True)
        # detect real images
        fake_output = discriminator(generated_images, training=True)

        # Compute the losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Obtain the current gradients of the two NNs
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Fix the gradients with the optimizer and update the NNs.
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# display function
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('img/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# Training loop that also shows images as they are being produced and improved
def train(generator, generator_optimizer, discriminator, discriminator_optimizer, dataset, epochs,
          checkpoint, checkpoint_prefix, seed, batch_size, noise_dim):

    for epoch in range(epochs):
        print("epoch : " , epoch+1)
        start = time.time()

        img_batch_count = 0
        for image_batch in dataset:
            if ((img_batch_count % 10) == 0):
                print("img_batch_count: " , img_batch_count)
            img_batch_count = img_batch_count + 1
            train_step(generator, generator_optimizer, discriminator, discriminator_optimizer, image_batch,
                       batch_size, noise_dim)

        # Produce images for the GIF as we go
        #display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        generate_and_save_images(generator, epochs, seed)
    return generator, discriminator

print("GAN Start")

# load the MNIST Data - data set for  digits recognition
(train_images, train_labels), (_ , _) = tf.keras.datasets.mnist.load_data()  # in python _ is a don't care variable

#Print image shapa
print("train_images shape: ", train_images.shape)  # (6000 samples of 28x28 grayscale images)

# Reshape image from 6000 x 28 x 28 to 6000 x 28 x 28 x 1. This makes images have 1 channel and becom ready
# for processing with convolutional NN. Also, they are casted floating point values
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# Normalize the image pixels to the range [-1.0, 1.0]
train_images = (train_images - 127.5)/127.5    # not clear why this is not 128, perhaps rounding issues

# Create batches and shuffle data to randomly mix everything and prevent NN to memory order of elements
BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Generator GAB
generator = make_generator_model()
generator.summary()

# create test image
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Show image
#plt.imshow(generated_image[0, :, : , 0], cmap = 'gray')
#plt.show()


# test the discriminator
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print("Discriminator decision: ", decision)

# create optimizers - Adam optimization
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Setup training loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

#setup checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


discriminator, generator = train(generator, generator_optimizer, discriminator, discriminator_optimizer,
                                 train_dataset, EPOCHS, checkpoint, checkpoint_prefix, seed, BATCH_SIZE, noise_dim)
print("GAN End.")