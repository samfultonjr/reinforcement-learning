const {DQNAgent} = require('./algorithims/DQN');
const {GAAgent} = require('./algorithims/GA');

const DQN = (hyperParams)=>{return new DQNAgent(hyperParams)}
const GA = (hyperParams)=>{return new GAAgent(hyperParams)}

module.exports = {
    DQN,
    GA
}