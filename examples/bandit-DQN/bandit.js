const rl = require('../..//index');
const tf = require('@tensorflow/tfjs');
let step = 0;

let arch = [
    {inputShape: 1, units: 14, activation: 'relu'},
    {units: 2, activation: 'softmax'}
];


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


function getState(){
    return [0];
}

function calcReward(state, action){
    let episodeDone = false;
    step++;
    if(step === 100){episodeDone = true; step = 0;}
    if(action === 1)return {reward: 1, newState:[0], done: true, episodeDone};
    else{ return {reward: 0, newState:[0], done: true, episodeDone}}
}