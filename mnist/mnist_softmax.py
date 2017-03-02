import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.session_bundle import exporter


def main(flags):
    mnist = input_data.read_data_sets(flags.mnist_data_dir, reshape=False, one_hot=True)

    # neural network with 1 layer of 10 softmax neurons
    #
    # · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
    # \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
    #   · · · · · · · ·                                              Y [batch, 10]

    # The model is:
    #
    # Y = softmax( X * W + b)
    #              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
    #              W: weight matrix with 784 lines and 10 columns
    #              b: bias vector with 10 dimensions
    #              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
    #              softmax(matrix) applies softmax on each line
    #              softmax(line) applies an exp to each value then divides by the norm of the resulting line
    #              Y: output matrix with 100 lines and 10 columns

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 10])
    # weights W[784, 10]   784=28*28
    W = tf.Variable(tf.zeros([784, 10]))
    # biases b[10]
    b = tf.Variable(tf.zeros([10]))

    # flatten the images into a single line of pixels
    # -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
    XX = tf.reshape(X, [-1, 784])

    # The model
    Y = tf.nn.softmax(tf.matmul(XX, W) + b)

    # loss function: cross-entropy = - sum( Y_i * log(Yi) )
    #                           Y: the computed output vector
    #                           Y_: the desired output vector

    # cross-entropy
    # log takes the log of each element, * multiplies the tensors element by element
    # reduce_mean will add all the components in the tensor
    # so here we end up with the total cross-entropy for all images in the batch
    cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
    # *10 because  "mean" included an unwanted division by 10

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(flags.iterations):
        batch_X, batch_Y = mnist.train.next_batch(flags.batch_size)

        # the backpropagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

        print("Iteration:" + str(i) + "/" + str(flags.iterations))

    print("******* Training is Done! *******")

    print("Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

    # Export model to Tensorflow Serving
    saver = tf.train.Saver()

    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': X}),
            'outputs': exporter.generic_signature({'y': Y})
        }
    )
    model_exporter.export(flags.model_dir, tf.constant(flags.model_version), sess)
    print("Model Saved at", str(flags.model_dir), "version:", flags.model_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=2000,
        help='Number of iterations to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--mnist_data_dir',
        type=str,
        default='./MNIST_data',
        help='Directory to put the mnist data.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='Directory to export model.'
    )
    parser.add_argument(
        '--model_version',
        type=int,
        default=1,
        help='Model version'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
