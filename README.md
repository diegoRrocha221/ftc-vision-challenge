<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README</title>
</head>
<body>

<h1>Image Classification with CNN using TensorFlow</h1>

<p>This project uses a Convolutional Neural Network (CNN) implemented in TensorFlow to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images of 32x32 pixels, divided into 10 classes.</p>

<h2>1. Loading and Preparing the Dataset</h2>
<p>The CIFAR-10 dataset is loaded directly from TensorFlow. The images are normalized (values from 0 to 255 are scaled to 0 to 1) to improve the neural network's performance. The classes present in the dataset are:</p>
<ul>
    <li>Airplane</li>
    <li>Car</li>
    <li>Bird</li>
    <li>Cat</li>
    <li>Deer</li>
    <li>Dog</li>
    <li>Frog</li>
    <li>Horse</li>
    <li>Ship</li>
    <li>Truck</li>
</ul>
<p>Example images from the training set are visualized to ensure the data is loaded correctly.</p>

<h2>2. Building the CNN Model</h2>
<p>The model is built using Keras (a high-level API in TensorFlow). The model's architecture includes:</p>
<ul>
    <li><strong>Convolutional Layers:</strong> 3 convolutional layers with 32, 64, and 64 filters, respectively. Each layer uses the ReLU activation function.</li>
    <li><strong>Pooling Layers:</strong> 2 pooling layers that reduce the image dimensions by half.</li>
    <li><strong>Flatten Layer:</strong> Converts the 3D outputs of the last convolutional layer into a single dimension.</li>
    <li><strong>Dense Layer:</strong> 64 neurons with the ReLU activation function.</li>
    <li><strong>Output Layer:</strong> 10 neurons (one for each class) without an activation function (logits).</li>
</ul>
<p>The model is compiled with the Adam optimizer and the Sparse Categorical Crossentropy loss function.</p>

<h2>3. Training the Model</h2>
<p>The model is trained for 10 epochs using the training set. Validation is performed at each epoch using the test set.</p>

<h2>4. Evaluating the Model</h2>
<p>After training, the model is evaluated on the test set to measure its accuracy. The accuracy is displayed as a performance metric.</p>

<h2>5. Visualizing Predictions</h2>
<p>The model's predictions are visualized for a few images from the test set. Blue indicates a correct prediction, while red indicates an incorrect prediction. The probability for each class is also visualized as a bar chart.</p>

<h2>Requirements</h2>
<p>To run this project, you will need the following libraries:</p>
<ul>
    <li>TensorFlow</li>
    <li>NumPy</li>
    <li>Matplotlib</li>
</ul>
<p>You can install the dependencies with the following command:</p>

<pre><code>pip install tensorflow numpy matplotlib</code></pre>

<h2>Running the Code</h2>
<p>To execute the code, run the Python script:</p>

<pre><code>python filename.py</code></pre>

<h2>Contributions</h2>
<p>Contributions to improve this project are welcome. Feel free to open issues and pull requests.</p>

</body>
</html>
