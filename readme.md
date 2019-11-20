# reinforcement-learning

Easy reinforcement learning using tensorflow.js


* [X] Deep Q Network (DQN)
* [ ] Genetic Algorithim (GA)
* [ ] Examples
* [ ] Training Dashboard (Tensorboard)

## Node

```javascript
const rl = require('reinforcement-learning');
```


## Browser

You will need to include reinforcement-learning as well as tensorflow.js
```html
<script src="https://unpkg.com/reinforcement-learning/index.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tensorflow/1.1.2/tf.min.js"></script>
```

## DQN

### Parameters
        arche
        epsilon
        epsilonDecay
        replayMemorySize
        miniBatchSize
        actionSpaceSize
        minReplaySize
        updateTargetEvery
        accuracyLookbackSize

```javascript
const rl = require('reinforcement-learning');
let step = 0;

let arch = [
    {inputShape: 1, units: 14, activation: 'relu'},
    {units: 2, activation: 'softmax'}
];

function getState(){return [0];}

function calcReward(state, action){
    // Every 100 steps end the episode
    step++;
    let episodeDone = false;
    if(step === 100){episodeDone = true; step = 0;}

    // Two armed bandit. Agent has to learn to always pick 1
    if(action === 1)return {reward: 1, newState:[0], done: true, episodeDone};
    else{ return {reward: 0, newState:[0], done: true, episodeDone}}
}


(async()=>{
    let agent = rl.DQN({
        arch, 
        epsilon: 0, 
        epsilonDecay: 0, 
        replayMemorySize: 1000, 
        miniBatchSize: 64, 
        actionSpaceSize: 2, 
        minReplaySize: 100, 
        updateTargetEvery: 1,
        accuracyLookbackSize: 500
    });

    await agent.learn({accuracy: 95, getState, calcReward});

})();



```

## evoConfig
evoConfig is a required object containing the attributes of all neural networks that reinforcement-learning will create. **Inputs** will simply be the number of inputs you will give the network. **Hidden** will be an array of objects which each contain the amount of neurons that layer will be given along with the activation function used. A list of the available activation functions and an explanation as to which ones you should use can be found [here](#activation-functions). **Output** is an object defining the output layer. Just like the hidden layers, you will define the number of neurons as well as the activation function. 

## getState()
getState() needs to be defined in order to let the agents get their current environment state (inputs). In the example above getState() gets the players position off of the DOM as well as the opponents position and then returns them in an array. These values should be numbers between 0 and 1 and need to be returned in an array. In order to get your inputs in between 0 and 1 it is common practice to divide all inputs by the highest possible value for that input. The amount of values returned from this function should match the number of inputs specified in evoConfig. You will never have to call getState(), it will be automatically called when exploring and evolving.

## makeMove(prediction)
makeMove() needs to be defined in order to execute upon the predicted move. In the example above there are 2 outputs, therefore the length of the outputs array given to makeMove will be 2. This is used for classification. In our example if outputs[0] is greater than outputs[1] then make a move such as turn left, if outputs[1] is greater than outputs[0] then make a move such as turn right. If you have more than 2 outputs, the same principles apply, the highest value in the array decides the move. You will never have to call makeMove(), this will automatically be called when exploring and evolving. Once you have made an action based on the outputs return a boolean with true if the player is still alive and flase if the player has died. 

## explore(epochs)
Explore will create neural networks with the structure you defined in evoConfig and give the network random weights. This random network will then get the state (inputs) using getState() and then make a decision upon that state (outputs) and make a move using makeMove(). This will be repeated for the amount of epochs you specify. The purpose of explore is to find a network that does slightly better than the others. The network with the highest score will then be used as a base model to provide a template for the networks being created within evolve() to mutate off of. Every time a network gets a new highscore it replace the base model currently being used.

## evolve(epochs)
Evolve make a clone of the network that has done the best so far (previously referred to as the base model) and mutate the weights in order to make a slightly different network. This network will then take in the current state (inputs) using the getState() function and make a move using makeMove() based on the prediction from the current state. The goal is to get a network that performs better than the base model and replace the base model and constantly evolve that base model. This repeat for the specified amount of epochs. If no amount of epochs is specified it will run indefinitely. 

## Activation Functions
The available activation functions are **'elu'**, **'hardSigmoid'**, **'linear'**, **'relu'**, **'relu6'**, **'selu'**, **'sigmoid'**, **'softmax'**, **'softplus'**, **'softsign'** and **'tanh'**. For most situations it is recommended that you use 'relu' for the hidden layers and for the output if you are solving a classification problem (turn left || turn right) (red || blue) use the 'softmax' activation function. For a more detailed explanation on which activation functions you should use watch Saraj Raval's video [Which Activation Function Should I Use?](https://www.youtube.com/watch?v=-7scQpJT7uo).


## License
[MIT](https://choosealicense.com/licenses/mit/)
