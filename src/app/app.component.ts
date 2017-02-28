import { Component, OnInit } from '@angular/core';
import { NeuralNetwork } from './neural-network/neural-network';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'app works!';
  nn = new NeuralNetwork();

  ngOnInit() { debugger
    // console.log(this.nn.dot([2, 4, 1], [2, 2, 3]));
    // console.log(this.nn.sigmoid([0, 2, 3]));

    // # The training set. We have 4 examples, each consisting of 3 input values
    // # and 1 output value.
    const training_set_inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]];
    const training_set_outputs = [0, 1, 1, 0];
    this.nn.train(training_set_inputs, training_set_outputs, 1000);
    const res = this.nn.think([[1, 1, 1]]);
    console.log('result = ' + res);
  }

}
