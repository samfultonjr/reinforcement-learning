
function importBestTF(){
    try{tf = require('@tensorflow/tfjs-node-gpu');}
    catch{
        try{tf = require('@tensorflow/tfjs-node');}
        catch{tf = require('@tensorflow/tfjs');}
    }
}

module.exports = {
    importBestTF
}