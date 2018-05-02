import * as tf from '@tensorflow/tfjs';
/**
 * The web camera class that is used to capture image.
 * Goolge's Tensorflow js example code is being used.
*/
class Webcam {
  /**
 * Crops an image tensor so we get a square image with no white space.
 * @param {Tensor4D} img An input image Tensor to crop.
 * @return {Tensor4D} return a croped Tensor4D.
 */
  static cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  /**
   * Setup a element to display the image.
   * @param {object} element is a dom element in the html page.
   * @return {null} No return.
   */
  constructor(element) {
    this.webcamElement = element;
  }

  /**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * @return {number} Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
  capture() {
    return tf.tidy(() => {
    // Reads the image as a Tensor from the webcam <video> element.
      const webcamImage = tf.fromPixels(this.webcamElement);
      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = Webcam.cropImage(webcamImage);

      // Expand the outer most dimension so we have a batch size of 1.
      const batchedImage = croppedImage.expandDims(0);
      // Normalize the image between -1 and 1. The image comes in between 0-255,
      // so we divide by 127 and subtract 1.
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

  /**
 * Adjusts the video size so we can make a centered square crop without
 * including whitespace.
 * @param {number} width The real width of the video element.
 * @param {number} height The real height of the video element.
 * @return {null} No return.
 */
  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  /**
   * Initial and setup the camera.
   * @return {object} Return a promise after initailizing success or fail
   */
  async setup() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia ||
        navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          stream => {
            this.webcamElement.srcObject = stream;
            this.webcamElement.addEventListener('loadeddata', async () => {
              this.adjustVideoSize(
                this.webcamElement.videoWidth,
                this.webcamElement.videoHeight
              );
              resolve();
            }, false);
          },
          error => {
            document.querySelector('#no-webcam').style.display = 'block';
          }
        );
      } else reject();
    });
  }
}
export default Webcam;
