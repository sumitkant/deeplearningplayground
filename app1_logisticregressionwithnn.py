import streamlit as st
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('datasets/app1/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/app1/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = np.float(0)
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -np.sum((Y)*np.log(A) + (1-Y)*np.log(1-A))/m
    dw = np.dot(X,(A - Y).T)/m
    db = np.sum(A-Y)/m
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction
@st.cache
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params['w']
    b = params['b']
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train =predict(w,b,X_train)

    # Print train/test Errors
    if print_cost:
        # print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100,
        # print("test accuracy: {} %".format())
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

    d = {"costs": costs,
         "train_accuracy": train_accuracy,
         "test_accuracy": test_accuracy,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def app():

    st.sidebar.title('Model Parameters')
    st.markdown('''
    # Logistic Regression with Neural Network Mindset
    
    Build a cat recognizer with Logistic Regression from scratch
    
    ## Importing Libraries 
    
    ```python
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import h5py
    import scipy
    from PIL import Image
    from scipy import ndimage
    ```
    
    ## Dataset
    Each observation is a `64 x 64` image, that is flattened for training and rescaled by dividing by 255 
    ''')

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    st.markdown("* Train X shape: " + str(train_set_x_flatten.shape) + '\n'
                "* Train Y shape: " + str(train_set_y.shape) + '\n'
                "* Test X shape: " + str(test_set_x_flatten.shape) + '\n'
                "* Test Y shape: " + str(test_set_y.shape))

    index = st.sidebar.slider('Training Image Index', 0, train_set_x_orig.shape[0])
    st.write('Image Shape:', train_set_x_orig[index].shape)
    im = Image.fromarray(train_set_x_orig[index])
    st.image(im, width=200)
    st.write("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")

    st.markdown('''
    
    ## Helper Functions
    
    ### 1. Sigmoid Function
    ```python
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    ```
    ### 2. Initialize Parameters
    ```python
    def initialize_with_zeros(dim):
        w = np.zeros([dim,1])
        b = np.float(0)
        return w, b
    ```
    
    ### 3. Forward Propogation
    
    ```python
    def propagate(w, b, X, Y):
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        cost = -np.sum((Y)*np.log(A) + (1-Y)*np.log(1-A))/m
        dw = np.dot(X,(A - Y).T)/m
        db = np.sum(A-Y)/m
        cost = np.squeeze(np.array(cost))
        grads = {"dw": dw, "db": db}
        return grads, cost
    ```
    
    ### 4. Optimizer
    ```python
    def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
        costs = []
        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            # update rule (≈ 2 lines of code)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        return params, grads, costs
    ```
    
    ### 5. Predict
    
    ```python
    def predict(w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        # Compute vector "A" predicting the probabilities of a cat being present in the picture        
        A = sigmoid(np.dot(w.T,X) + b)
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0 
        return Y_prediction
    ```
    ### 6. Model
    
    ```python
    def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        w,b = initialize_with_zeros(X_train.shape[0])
        params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = params['w']
        b = params['b']
        Y_prediction_test = predict(w,b,X_test)
        Y_prediction_train =predict(w,b,X_train)
        
        # Print train/test Errors
        if print_cost:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d
    ```
    ## Modelling
    ''')

    num_iterations = st.sidebar.slider('Iterations', 10, 2000, 200, step=10)
    learning_rate = st.sidebar.slider('Learning Rate', 0.001, 0.05, 0.005, step=0.001)
    # print_cost = st.selectbox('print_cost :', [True, False])
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations,
                                      learning_rate, True)

    st.write('Train Accuracy (%) :', np.round(logistic_regression_model['train_accuracy'][0],2))
    st.write('Test Accuracy (%) :', np.round(logistic_regression_model['test_accuracy'],2))

    st.markdown('''
    ## Predictions
    ''')

    # Example of a picture that was wrongly classified.
    num_px = train_set_x_orig.shape[1]
    index2 = st.sidebar.slider('Test Image Index', 0, test_set_x_orig.shape[0], 10)
    st.image(Image.fromarray(test_set_x_orig[index2]), width=200)
    st.write("y = " + str(test_set_y[0, index2]) + ", you predicted that it is a \"" + classes[
        int(logistic_regression_model['Y_prediction_test'][0, index2])].decode("utf-8") + "\" picture.")

    st.markdown('''
    ## Costs
    ''')

    costs = np.squeeze(logistic_regression_model['costs'])
    f, ax = plt.subplots(1, 1, figsize=(6,2))
    ax.plot(costs)
    ax.set_ylabel('cost')
    ax.set_xlabel('iterations (per hundreds)')
    ax.set_title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    st.pyplot(f)

    st.markdown('''
        ## Test with your own image 
        Go break some eggs!
        ''')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((num_px, num_px))
        st.image(image, width=200)
        image = np.array(image)/255.
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)
        st.write("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
            int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")