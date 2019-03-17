import tensorflow as tf
import numpy as np

learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.

x_dataset = np.linspace(-1, 1, 100)

num_coeffs = 9
y_dataset_params = [0, ] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3


def model(X, y):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


def split_dataset(x_data_set, y_data_set, ratio):
    arr = np.arange(x_data_set.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_data_set.size)
    x_train_set = x_data_set[arr[0:num_train]]
    x_test_set = x_data_set[arr[num_train:x_data_set.size]]
    y_train_set = y_data_set[arr[0:num_train]]
    y_test_set = y_data_set[arr[num_train:x_data_set.size]]
    return x_train_set, x_test_set, y_train_set, y_test_set


(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)
print(tf.reduce_sum(tf.square(Y - y_model)))
print(reg_lambda)
print(tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w))))
reduce = tf.reduce_sum(tf.square(w))
multi = tf.multiply(reg_lambda, reduce)
reduce_sum = tf.reduce_sum(tf.square(Y - y_model))
cost = tf.divide(reduce, multi)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for reg_lambda in np.linspace(0, 1, 100):
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
    final_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
    print("final cost pre ", final_cost)
    print("reg lambda", reg_lambda)

print("final cost", final_cost)

sess.close()
exit()