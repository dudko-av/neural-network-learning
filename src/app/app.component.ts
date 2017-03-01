import { Component, OnInit } from '@angular/core';
import { NeuralNetwork } from './neural-network/neural-network';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'app works!';
  nn: NeuralNetwork;

  ngOnInit() { debugger
    const xorInput: number[][] = [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0]
    ];

    const xorIdeal: number[][] = [
      [0.0], [1.0], [1.0], [0.0]
    ];

    // System.out.println("Learn:");
    console.log('Learn:');

    this.nn = new NeuralNetwork(2, 3, 1, 0.7, 0.9);

    /*    NumberFormat percentFormat = NumberFormat.getPercentInstance();
     percentFormat.setMinimumFractionDigits(4);*/


    for (let i = 0; i < 10000; i++) {
      for (let j = 0; j < xorInput.length; j++) {
        this.nn.computeOutputs(xorInput[j]);
        this.nn.calcError(xorIdeal[j]);
        this.nn.learn();
      }
      /*      System.out.println( "Trial #" + i + ",Error:" +
       percentFormat .format(network.getError(xorInput.length)) );*/
      console.log(
        'Trial #' + i + ', Error: ' +
        this.nn.getError(xorInput.length)
      );
    }

    // System.out.println("Recall:");
    console.log('Recall:');

    for (let i = 0; i < xorInput.length; i++) {

      for (let j = 0; j < xorInput[0].length; j++) {
        // System.out.print( xorInput[i][j] +":" );
        console.log(xorInput[i][j] + ' : ');
      }

      const out: number[] = this.nn.computeOutputs(xorInput[i]);
      // System.out.println("="+out[0]);
      console.log(' = ' + out[0]);
    }
  }

}
