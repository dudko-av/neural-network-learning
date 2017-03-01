/*
 http://www.informit.com/articles/article.aspx?p=30596&seqNum=6
 */

declare const math: any;

export class NeuralNetwork {
  /**
   * The global error for the training.
   */
  protected globalError: number;

  /**
   * The number of input neurons.
   */
  protected inputCount: number;

  /**
   * The number of hidden neurons.
   */
  protected hiddenCount: number;

  /**
   * The number of output neurons
   */
  protected outputCount: number;

  /**
   * The total number of neurons in the network.
   */
  protected neuronCount: number;

  /**
   * The number of weights in the network.
   */
  protected weightCount: number;

  /**
   * The learning rate.
   */
  protected learnRate: number;

  /**
   * The outputs from the various levels.
   */
  protected fire: number[];

  /**
   * The weight matrix this, along with the thresholds can be
   * thought of as the "memory" of the neural network.
   */
  protected matrix: number[];

  /**
   * The errors from the last calculation.
   */
  protected error: number[];

  /**
   * Accumulates matrix delta's for training.
   */
  protected accMatrixDelta: number[];

  /**
   * The thresholds, this value, along with the weight matrix
   * can be thought of as the memory of the neural network.
   */
  protected thresholds: number[];

  /**
   * The changes that should be applied to the weight
   * matrix.
   */
  protected matrixDelta: number[];

  /**
   * The accumulation of the threshold deltas.
   */
  protected accThresholdDelta: number[];

  /**
   * The threshold deltas.
   */
  protected thresholdDelta: number[];

  /**
   * The momentum for training.
   */
  protected momentum: number;

  /**
   * The changes in the errors.
   */
  protected errorDelta: number[];

  constructor(
    inputCount: number, // inputCount — The number of neurons that are in the input layer.
    hiddenCount: number, // hiddenCount — The number of neurons that are in the hidden layer.
    outputCount: number, // outputCount — The number of neurons that are in the output layer.
    learnRate: number, // learnRate — The learning rate for the backpropagation training algorithm.
    momentum: number // momentum — The momentum for the backpropagation training algorithm.
  ) {
    this.inputCount = inputCount;
    this.hiddenCount = hiddenCount;
    this.outputCount = outputCount;
    this.learnRate = learnRate;
    this.momentum = momentum;

    this.neuronCount = inputCount + hiddenCount + outputCount;
    this.weightCount = (inputCount * hiddenCount) + (hiddenCount * outputCount);
    this.fire = new Array(this.neuronCount);
    this.matrix = new Array(this.weightCount);
    this.matrixDelta = new Array(this.weightCount);
    this.thresholds = new Array(this.neuronCount);
    this.errorDelta = new Array(this.neuronCount);
    this.error = new Array(this.neuronCount);
    this.accThresholdDelta = new Array(this.neuronCount);
    this.accMatrixDelta = new Array(this.weightCount);
    this.thresholdDelta = new Array(this.neuronCount);

    this.reset();
  }

  /**
   * Returns the root mean square error for a complet training set.
   *
   * @param len The length of a complete training set.
   * @return The current error for the neural network.
   */
  getError(len: number): number {
    const err = Math.sqrt(this.globalError / (len * this.outputCount));
    this.globalError = 0; // clear the accumulator
    return err;
  }

  /**
   * The threshold method. You may wish to override this class to provide other
   * threshold methods.
   *
   * @param sum The activation from the neuron.
   * @return The activation applied to the threshold method.
   */
  threshold(sum: number): number {
    return 1 / (1 + Math.exp(-1 * sum));
  }

  /**
   * Compute the output for a given input to the neural network.
   *
   * @param input The input provide to the neural network.
   * @return The results from the output neurons.
   */
  computeOutputs(input: number[]): number[] {
    let i, j;
    const hiddenIndex = this.inputCount;
    const outIndex = this.inputCount + this.hiddenCount;

    for (i = 0; i < this.inputCount; i++) {
      this.fire[i] = input[i];
    }

    // first layer
    let inx = 0;

    for (i = hiddenIndex; i < outIndex; i++) {
      let sum = this.thresholds[i];

      for (j = 0; j < this.inputCount; j++) {
        sum += this.fire[j] * this.matrix[inx++];
      }
      this.fire[i] = this.threshold(sum);
    }

    // hidden layer
    const result: number[] = new Array(this.outputCount);

    for (i = outIndex; i < this.neuronCount; i++) {
      let sum = this.thresholds[i];

      for (j = hiddenIndex; j < outIndex; j++) {
        sum += this.fire[j] * this.matrix[inx++];
      }
      this.fire[i] = this.threshold(sum);
      result[i - outIndex] = this.fire[i];
    }

    return result;
  }

  /**
   * Calculate the error for the recogntion just done.
   *
   * @param ideal What the output neurons should have yielded.
   */
  calcError(ideal: number[]) {
    let i, j;
    const hiddenIndex = this.inputCount;
    const outputIndex = this.inputCount + this.hiddenCount;

    // clear hidden layer errors
    for (i = this.inputCount; i < this.neuronCount; i++) {
      this.error[i] = 0;
    }

    // layer errors and deltas for output layer
    for (i = outputIndex; i < this.neuronCount; i++) {
      this.error[i] = ideal[i - outputIndex] - this.fire[i];
      this.globalError += this.error[i] * this.error[i];
      this.errorDelta[i] = this.error[i] * this.fire[i] * (1 - this.fire[i]);
    }

    // hidden layer errors
    let winx = this.inputCount * this.hiddenCount;

    for (i = outputIndex; i < this.neuronCount; i++) {
      for (j = hiddenIndex; j < outputIndex; j++) {
        this.accMatrixDelta[winx] += this.errorDelta[i] * this.fire[j];
        this.error[j] += this.matrix[winx] * this.errorDelta[i];
        winx++;
      }
      this.accThresholdDelta[i] += this.errorDelta[i];
    }

    // hidden layer deltas
    for (i = hiddenIndex; i < outputIndex; i++) {
      this.errorDelta[i] = this.error[i] * this.fire[i] * (1 - this.fire[i]);
    }

    // input layer errors
    winx = 0; // offset into weight array
    for (i = hiddenIndex; i < outputIndex; i++) {
      for (j = 0; j < hiddenIndex; j++) {
        this.accMatrixDelta[winx] += this.errorDelta[i] * this.fire[j];
        this.error[j] += this.matrix[winx] * this.errorDelta[i];
        winx++;
      }
      this.accThresholdDelta[i] += this.errorDelta[i];
    }
  }

  /**
   * Modify the weight matrix and thresholds based on the last call to
   * calcError.
   */
  public learn() {
    let i;

    // process the matrix
    for (i = 0; i < this.matrix.length; i++) {
      this.matrixDelta[i] = (this.learnRate * this.accMatrixDelta[i]) + (this.momentum * this.matrixDelta[i]);
      this.matrix[i] += this.matrixDelta[i];
      this.accMatrixDelta[i] = 0;
    }

    // process the thresholds
    for (i = this.inputCount; i < this.neuronCount; i++) {
      this.thresholdDelta[i] = this.learnRate * this.accThresholdDelta[i] + (this.momentum * this.thresholdDelta[i]);
      this.thresholds[i] += this.thresholdDelta[i];
      this.accThresholdDelta[i] = 0;
    }
  }

  /**
   * Reset the weight matrix and the thresholds.
   */
  reset() {
    let i;
    for (i = 0; i < this.neuronCount; i++) {
      this.thresholds[i] = 0.5 - (Math.random());
      this.thresholdDelta[i] = 0;
      this.accThresholdDelta[i] = 0;
    }
    for (i = 0; i < this.matrix.length; i++) {
      this.matrix[i] = 0.5 - (Math.random());
      this.matrixDelta[i] = 0;
      this.accMatrixDelta[i] = 0;
    }
  }

}
