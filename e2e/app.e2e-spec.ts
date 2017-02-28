import { NeuralNetworkLearningPage } from './app.po';

describe('neural-network-learning App', function() {
  let page: NeuralNetworkLearningPage;

  beforeEach(() => {
    page = new NeuralNetworkLearningPage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
