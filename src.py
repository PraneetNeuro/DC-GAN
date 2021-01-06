import tensorflow as tf
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, path_of_dataset, buffer_size=60000, batch_size=256, size=(28, 28)):
        self.path_of_dataset = path_of_dataset
        self.size = size
        self.X = []
        self.Y = []
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.populate_dataset()
        self.make_dataset()

    def populate_dataset(self):
        for directory in os.listdir(self.path_of_dataset):
            if directory.startswith('.'):
                continue
            for img in os.listdir(os.path.join(self.path_of_dataset, directory)):
                try:
                    img = cv2.imread(os.path.join(self.path_of_dataset, directory, img))
                    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
                    img = cv2.resize(img, self.size)
                    img = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
                    self.X.append(img)
                    self.Y.append(directory)
                except:
                    pass

    def make_dataset(self):
        self.X = tf.convert_to_tensor(np.array(self.X), dtype=tf.float32)
        self.X = tf.data.Dataset.from_tensor_slices(self.X).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)


class GAN:
    def __init__(self, path_of_dataset, epochs=50, noise_dim=100, num_examples=16):
        self.dataset = Dataset(path_of_dataset)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.EPOCHS = epochs
        self.noise_dim = noise_dim
        '''
        NOISE DIMENSION denotes the shape of the noise that is given as input to the generator,
        affects the quality of the generated image directly,
        100 as the noise dimension is the convention for smaller models
        '''
        self.num_examples_to_generate = num_examples

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        '''
        SEED has the random noise samples for generating new images which when given as input to the generator
        '''
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.train(self.dataset.X, self.EPOCHS)

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        '''
        Dense layer with number of units required to reshape it into a matrix that can be upsampled through
        Transpose Convolution layers which learns filters of the the target domain 
        '''
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)

        model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                  activation='sigmoid'))
        assert model.output_shape == (None, 28, 28, 3)
        return model

    def make_discriminator_model(self):
        '''
        The discriminator model is a vanilla convolutional neural network image classifier
        classes: REAL / FAKE (Binary classifier)
        '''
        model = tf.keras.Sequential()
        '''
        Input shape should be the shape of the generated image which in turn should be of the same size as a sample
        from the training data
        '''
        model.add(tf.keras.layers.Input([28, 28, 3]))
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        '''
        Tensor should be flattened before being passed to the Dense layer
        '''
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        '''
        1 unit in the final Dense layer for the discriminator expressing the probability of the image being
        fake (0-0.4) or real (0.5-1)
        '''
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        '''
        REAL LOSS denotes the loss when the discriminator predicts whether the given image which is real as 
        real or fake
        When optimised, it makes the discriminator identify the real images
        '''
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        '''
        FAKE LOSS denotes the loss when the discriminator predicts whether the given image which is fake (genereated by the
        generator) as real or fake
        When optimised, it makes the discriminator identify the fake images (generated by the generator)
        '''
        total_loss = real_loss + fake_loss
        '''
        TOTAL LOSS when optimised makes the discriminator identify the real images and fake images apart from each other
        This forces the generator to generate images which fool the discriminator into classifying them as real images
        '''
        return total_loss

    def generator_loss(self, fake_output):
        '''
        GENERATOR LOSS when optimised makes the generator generate more realistic images
        '''
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        '''
        Marking the function as a tf.function causes the function to be compiled as a tensorflow function, making it
        faster and compatible with eager execution
        '''
        noise = tf.random.normal([self.dataset.BATCH_SIZE, self.noise_dim])
        '''
        Batch size number of random values in the shape (100,) used as input to the generator for training
        '''
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            '''
            Using GradientTape function allows defining a customised fit function as the keras fit function is not 
            suitable for training a GAN
            Can be used to track computations and calculate derivatives with respect to a variable
            '''
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
        '''
        Custom training loop
        '''
        print('Training initiliased')
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            self.generate_and_save_images(self.generator,
                                          epoch + 1,
                                          self.seed)
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)
        '''
        Training is set to false when used for inference
        '''
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


GAN('<PATH OF DATASET>')
