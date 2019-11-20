
# reinforcement-learning

  

Easy reinforcement learning using tensorflow.js

  
  

*  [X] Deep Q Network (DQN)

* [ ] Genetic Algorithim (GA)

* [ ] Examples

* [ ] Training Dashboard (Tensorboard)

  

## Require

  

```javascript

const  rl = require('reinforcement-learning');

```

  
  


  

## DQN

  

### Parameters

 - ``arch`` - Architechture of the neural network

	

 - ``epsilon``	- % of actions that should be taken randomly for exploration  

 - ``epsilonDecay`` - Epsilon will be multiplied by this amount every episode

- ``replayMemorySize`` - Amount of previous steps left in memory to train on 

- ``miniBatchSize`` - Batch size to fit on 

- ``actionSpaceSize`` - Amount of possible actions the agent can take 

- ``minReplaySize`` - Minimum amount of memories allowed for fitting 

- ``updateTargetEvery``  How many episodes to wait to update the predictions network

- ``accuracyLookbackSize`` How many previous steps should be used to calculate accuracy

  

```javascript
const  rl = require('reinforcement-learning');
let  step = 0;


let  arch = [
{inputShape:  1, units:  14, activation:  'relu'},
{units:  2, activation:  'softmax'}
];
  
function  getState(){return [0];}

function  calcReward(state, action){
// Every 100 steps end the episode
step++;
let  episodeDone = false;
if(step === 100){episodeDone = true; step = 0;}

// Two armed bandit. Agent has to learn to always pick 1
if(action === 1)return {reward:  1, newState:[0], done:  true, episodeDone};
else{ return {reward:  0, newState:[0], done:  true, episodeDone}}
}


(async()=>{
let  agent = rl.DQN({
arch,
epsilon:  0,
epsilonDecay:  0,
replayMemorySize:  1000,
miniBatchSize:  64,
actionSpaceSize:  2,
minReplaySize:  100,
updateTargetEvery:  1,
accuracyLookbackSize:  500
});

  
await  agent.learn({accuracy:  95, getState, calcReward});

  
})();
  
  
  

```

  

  
  

## License

[MIT](https://choosealicense.com/licenses/mit/)