import * as tf from '@tensorflow/tfjs';

export let addExampleHandler;
let mouseDown = false;
const totals = [0, 0, 0, 0];
const CONTROLS = ['hhh', 'ttt', 'hht', 'tth'];

const hhhButton = document.getElementById('hhh');
const tttButton = document.getElementById('ttt');
const hhtButton = document.getElementById('hht');
const tthButton = document.getElementById('tth');
const thumbDisplayed = {};
const thumbCanvas = [
  document.getElementById(`${CONTROLS[0]}-thumb`),
  document.getElementById(`${CONTROLS[1]}-thumb`),
  document.getElementById(`${CONTROLS[2]}-thumb`),
  document.getElementById(`${CONTROLS[3]}-thumb`)
];
const trainTextDiv = document.getElementById('trainTextDiv');
const predictionText = document.getElementById('predictionText');
const trainProgress = document.getElementById('trainProgress');
const progressBar = document.getElementById('progressBar');
const loadMessageSpan = document.getElementById('loadMessageSpan');

export function setExampleHandler(_handler) {
  addExampleHandler = _handler;
}

async function handler(label) {
  mouseDown = true;
  const className = CONTROLS[label];
  // const button = document.getElementById(className);
  const total = document.getElementById(`${className}-total`);
  while (mouseDown) {
    addExampleHandler(label);
    // document.body.setAttribute('data-active', CONTROLS[label]);
    total.innerText = `${++totals[label]} examples`;
    await tf.nextFrame();
  }
  // document.body.removeAttribute('data-active');
}

hhhButton.addEventListener('mousedown', () => handler(0));
hhhButton.addEventListener('mouseup', () => { mouseDown = false; });

tttButton.addEventListener('mousedown', () => handler(1));
tttButton.addEventListener('mouseup', () => { mouseDown = false; });

hhtButton.addEventListener('mousedown', () => handler(2));
hhtButton.addEventListener('mouseup', () => { mouseDown = false; });

tthButton.addEventListener('mousedown', () => handler(3));
tthButton.addEventListener('mouseup', () => { mouseDown = false; });


export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[(i * 3) + 0] + 1) * 127;
    imageData.data[j + 1] = (data[(i * 3) + 1] + 1) * 127;
    imageData.data[j + 2] = (data[(i * 3) + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    // const thumbCanvas = document.getElementById(`${CONTROLS[label]}-thumb`);
    draw(img, thumbCanvas[label]);
  }
}

export function showLoadSccuess(text) {
  loadMessageSpan.innerText = text;
}

export function trainStatus(text) {
  trainTextDiv.innerText = text;
}

export function predictClass(classId) {
  predictionText.innerText = CONTROLS[classId];
}

export function showTrainProgress() {
  trainProgress.style.display = 'flex';
}

export function hideTrainProgress() {
  trainProgress.style.display = 'none';
}

export function epochStatus(currentEpochs, epochs) {
  const currentProcess = (currentEpochs + 1) / epochs;
  progressBar.style.width = `${currentProcess * 100}%`;
  progressBar.setAttribute('aria-valuenow', currentEpochs + 1);
  progressBar.innerText = `${currentEpochs + 1}/${epochs}`;
}
