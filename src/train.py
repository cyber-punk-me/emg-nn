import os
import shutil
import glob
from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder

data_path = str(Path("../data"))
print(os.listdir(data_path))
export_path_base = str(Path("../tf_export")) + os.sep

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')

allFiles = glob.glob(data_path + os.sep + "*.csv")
list = []
for file in allFiles:
    read = pd.read_csv(file, header = None)
    list.append(read)
df = pd.concat(list)

D = 64 # number of input features
M1 = 34 # first layer number of nodes, relatively arbitrarily chosen
M2 = 17 # second hidden layer number of nodes, relatively arbitrarily chosen
K = 4 # output layer nodes or number of classes

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.05)

N = len(Ytrain)
T = np.zeros((N, K))
for i in range(N):
    T[i, Ytrain[i]] = 1 # this creates an indicator/dummy variable matrix for the output layer. We need to do this for
# two reasons. 1) it creates an NxK matrix that will be broadcastable with the predictions generated from the forward
# function and used in the cost function. 2) when we argmax the predictions, it will turn into a matrix NxK of values only
# either 1 or 0 which can directly be compared with T to test the accuracy

def initialize_weights_and_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# using two hidden layers
def feed_forward(W3, W2, W1, b3, b2, b1, X):
    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    return tf.matmul(Z2, W3) + b3

tfX = tf.placeholder(tf.float32, [None, D], name='input') # creates placeholder variables without actually assigning values to them yet
tfY = tf.placeholder(tf.float32, [None, K], name='output') # None means it can take any size N total number of instances


W1 = initialize_weights_and_biases([D, M1])
W2 = initialize_weights_and_biases([M1, M2])
W3 = initialize_weights_and_biases([M2, K])
b1 = initialize_weights_and_biases([M1])
b2 = initialize_weights_and_biases([M2])
b3 = initialize_weights_and_biases([K])

pY_given_X = feed_forward(W3, W2, W1, b3, b2, b1, tfX)
y_max = tf.argmax(pY_given_X, dimension=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = tfY, logits = pY_given_X))

train_model = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_output = tf.argmax(pY_given_X, 1) # 1 refers to axis = 1, meaning it does argmax on each instance n

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)

for i in range(2000):
    session.run(train_model, feed_dict = {tfX: Xtrain, tfY: T})
    pred = session.run(predict_output, feed_dict = {tfX:Xtrain, tfY:T})
    if i % 250 == 0:
        print("classification_rate: {}".format(np.mean(Ytrain == pred)))

# Test Set evaluation
Ntest = len(Ytest)
Ttest = np.zeros((Ntest, K)) # test set indicator matrix
for i in range(Ntest):
    Ttest[i, Ytest[i]] = 1

predtest = session.run(predict_output, feed_dict = {tfX: Xtest, tfY: Ttest})
print("Test Set classification rate: {}".format(np.mean(Ytest == predtest))) # evaluates boolean as either 1 or 0 then

export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(tf.app.flags.FLAGS.model_version)))
if os.path.exists(export_path) and os.path.isdir(export_path):
    shutil.rmtree(export_path)

print("Exporting trained model to", export_path)

with session.graph.as_default():
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={
            "input": utils.build_tensor_info(tfX)
        },
        outputs={
            "output" : utils.build_tensor_info(y_max),
            "distr": utils.build_tensor_info(pY_given_X)
        },
        method_name=signature_constants.PREDICT_METHOD_NAME
    )
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": prediction_signature,
            "predict": prediction_signature
        })
    builder.save()
    
    converter = tf.contrib.lite.TFLiteConverter.from_session(session, [tfX], [pY_given_X, y_max])
    tflite_model = converter.convert()
    open(export_path_base + os.sep + "converted.tflite", "wb").write(tflite_model)
    
session.close()
