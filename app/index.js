import * as tf from '@tensorflow/tfjs';

import * as ui from './Ui';
import Webcam from './Webcam';
import { ControllerDataset } from './controller_dataset';

const LEARNING_RATE = 0.0001;
const BATCH_SIZE_FRACTION = 0.4;
const EPOCHS = 30;
const DENSE_UNITS = 100;
// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;
// let isPredicting = false;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenetA = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenetA.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenetA.inputs, outputs: layer.output });
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. HHH, TTT, HHT, TTH are
// labels 0, 1, 2, 3 respectively.
ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);

    // Draw the preview thumbnail.
    ui.drawThumb(img, label);
  });
});

/**
 * Sets up and trains the classifier.
 * @return {null} No return.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  ui.showTrainProgress();

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({ inputShape: [7, 7, 256] }),
      // Layer 1
      tf.layers.dense({
        units: DENSE_UNITS,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(LEARNING_RATE);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({ optimizer, loss: 'categoricalCrossentropy' });

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * BATCH_SIZE_FRACTION);
  if (!(batchSize > 0)) {
    throw new Error('Batch size is 0 or NaN. Please choose a non-zero fraction.');
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: EPOCHS,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus(`Loss: ${logs.loss.toFixed(5)}`);
        await tf.nextFrame();
      },
      onEpochEnd: epoch => {
        ui.epochStatus(epoch, EPOCHS);
        if (epoch + 1 === EPOCHS) ui.hideTrainProgress();
      }
    }
  });
}

async function predict() {
  const predictedClass = tf.tidy(() => {
    // Capture the frame from the webcam.
    const img = webcam.capture();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model.
    const activation = mobilenet.predict(img);

    // Make a prediction through our newly-trained model using the activation
    // from mobilenet as input.
    const predictions = model.predict(activation);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    return predictions.as1D().argMax();
  });

  const classId = (await predictedClass.data())[0];

  ui.predictClass(classId);
}

document.getElementById('trainBtn').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  // isPredicting = false;
  train();
});
document.getElementById('predictBtn').addEventListener('click', () => predict());

/**
 * Initial all element for the app.
 * @return {null} No return.
 */
async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
