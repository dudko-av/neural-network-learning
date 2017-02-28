declare const math: any;

export class NeuralNetwork {
  synapticWeights = [Math.random(), Math.random(), Math.random() * -1];
  counter = 0;

  constructor() {
    console.log(this.synapticWeights);
  }

  /*
   # The Sigmoid function, which describes an S shaped curve.
   # We pass the weighted sum of the inputs through this function to
   # normalise them between 0 and 1.
   */
  sigmoid(x: number[]) {
    return x.map(val => (1 / (1 + math.exp(-val))));
  }

  /*# The derivative of the Sigmoid function.
   # This is the gradient of the Sigmoid curve.
   # It indicates how confident we are about the existing weight.*/
  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  /*
   # We train the neural network through a process of trial and error.
   # Adjusting the synaptic weights each time.
   */
  train(trainingSetInputs: number[][], trainingSetOutputs: number[], numberOfTrainingIterations: number) { debugger
    for (let i = 0; i < numberOfTrainingIterations; i++) {

      // Pass the training set through our neural network (a single neuron).
      const output: number[] = this.think(trainingSetInputs);

      // Calculate the error (The difference between the desired output
      // and the predicted output).
      // const error = training_set_outputs - output
      const error: number[] = trainingSetOutputs
        .map((v, inx) => v - output[inx])
        .map((v, inx) => v * this.sigmoidDerivative(output[inx]));

      // Multiply the error by the input and again by the gradient of the Sigmoid curve.
      // This means less confident weights are adjusted more.
      // This means inputs, which are zero, do not cause changes to the weights.
      // adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
      const adjustment = trainingSetInputs[0].map((_, index) => {
        return trainingSetInputs.map(inputs => inputs[index]);
        // const res = adjustment.map(input => math.dot(vertical, error));
      }).map(inputs => math.dot(inputs, error));
      // console.log(adjustment);
      adjustment.forEach((input, inx) => {
        this.synapticWeights[inx] += input;
      });
      console.log('iteration = ' + i + ' - ' + this.synapticWeights);
    }
  }

  think(inputs: number[][]): number[] {
    // Pass inputs through our neural network (our single neuron).
    const res = inputs.map(input => math.dot(input, this.synapticWeights));
    return this.sigmoid(res);
  }

}
