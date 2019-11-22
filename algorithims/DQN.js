require('../src/importBestTF').importBestTF();

class DQNAgent {

    constructor({arch, epsilon, epsilonDecay, replayMemorySize, miniBatchSize, actionSpaceSize, maxReplayMemorySize, minReplaySize, updateTargetEvery, accuracyLookbackSize, discount}){
        this.arch = arch;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.replayMemorySize = replayMemorySize;
        this.miniBatchSize = miniBatchSize || 64;
        this.actionSpaceSize = actionSpaceSize;
        this.accuracy = 0;
        this.accuracyLookback = [];
        this.accuracyLookbackSize = accuracyLookbackSize;
        this.totalRewards = 0;
        this.totalSteps = 0;
        this.maxReplayMemorySize = maxReplayMemorySize;
        this.minReplaySize = minReplaySize;
        this.updateTargetEvery = updateTargetEvery;
        this.discount = discount;
        // Gets trained every step
        this.model = this.createModel();
        // Predicts against every step, gets updated after every episode to keep model stable
        this.targetModel = this.createModel();
        this.targetModel.setWeights(this.model.getWeights());
        this.replayMemory = [];
        this.targetUpdateCounter = 0;

        this.episodeSteps = 0;
        this.episode = 0;
        this.episodeStartTime = Date.now();
    }

    createModel(){
        const model = tf.sequential();
        for (let index = 0; index < this.arch.length; index++) {
            const layer = this.arch[index];
            model.add(tf.layers.dense(layer));
        };
        model.compile({optimizer:'adam', loss:'meanSquaredError'});
        return model;
    }

    updateReplayMemory(memory){
        if(this.replayMemory.length > this.maxReplayMemorySize){this.replayMemory.splice(Math.round(Math.random()*this.replayMemory.length), 1);}
        this.replayMemory.push(memory);
        if(this.accuracyLookback.length >= this.accuracyLookbackSize){this.accuracyLookback.shift()}
        this.accuracyLookback.push(memory.reward);
        this.totalSteps++;
        this.totalRewards += memory.reward;
    }

    getQs(state){
        return tf.tidy(()=>{
            return maxIndex(this.targetModel.predict(tf.tensor([state])).dataSync());
        });
    }

    chooseAction(state){
        if(Math.random() > this.epsilon){
            return this.getQs(state);
        }
        else{
            return Math.floor(Math.random() * this.actionSpaceSize);
        }
    }



    async train(terminalState){
        // let {X, Y} = tf.tidy( ()=>{ 
        if(this.replayMemory.length < this.minReplaySize){return}
        let miniBatch = sample(this.replayMemory, this.miniBatchSize);
        let currentStates = miniBatch.map((dp)=>{return dp.state});
        let currentQs = await this.model.predict(tf.tensor(currentStates)).array();
        let newCurrentStates = miniBatch.map((dp)=>{return dp.newState});
        let futureQs = await this.model.predict(tf.tensor(newCurrentStates)).array();

        let X = [];
        let Y = [];

        for (let index = 0; index < miniBatch.length; index++) {
            const {state, action, reward, newState, done} = miniBatch[index];
            let newQ;
            let currentQ;

            if(!done){
                let maxFutureQ = Math.max(futureQs);
                newQ = reward + (this.discount * maxFutureQ);
            }
            else{ newQ = reward }

            currentQ = currentQs[index];
            currentQ[action] = newQ;

            X.push(state);
            Y.push(currentQ);
        }

        await this.model.fit(tf.tensor(X), tf.tensor(Y), {batchSize: this.miniBatchSize});

        if(terminalState) this.targetUpdateCounter++;
        if(this.targetUpdateCounter >= this.updateTargetEvery){
            this.endEpisode();
        }
     
    }



    endEpisode(){
        this.targetModel.setWeights(this.model.getWeights());
        this.targetUpdateCounter = 0;
        // this.accuracy = (this.accuracyLookback.reduce(function(a, b) { return a + b; }, 0) / this.accuracyLookbackSize) * 100;
        // this.accuracy = (this.totalRewards / this.totalSteps) * 100;
        this.accuracy = 0;
        console.log(`Accuracy: ${this.accuracy}  Timing: ${(Date.now() - this.episodeStartTime)}  Epsilon: ${this.epsilon}`);
        this.epsilon = this.epsilon * this.epsilonDecay;
        this.episodeSteps = 0;
    }


    async learn({accuracy, getState, calcReward}){
        while(this.accuracy < accuracy){
            this.episodeSteps++;
            if(this.episodeSteps === 1){this.episode++; this.episodeStartTime = Date.now()}

            let state = await getState();
            let action = await this.chooseAction(state);
            let {reward, newState, done, episodeDone} = await calcReward(state, action);
            this.updateReplayMemory({state, action, reward, newState, done});
            await this.train(episodeDone);
        }
    }



}





function maxIndex(arr){
    return arr.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
}

function sample(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}



module.exports = {
    DQNAgent
}