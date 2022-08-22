const childProcess = require('child_process');
const base58 = require('base-58');
const Web3 = require('web3');


const address = "0x5B245bAdA3E41692F507e8a9eEf3107733a97BC0";
const privateKey = "2a25255ce1801666fd7c549cfb8e63dc54abab2d576f3b01cd653bc3060740cf";
const showCID = async()=>{
    const web3 = new Web3(new Web3.providers.HttpProvider("https://ropsten.infura.io/v3/0a55ff1dc9ff4d26999876d698f0b748"));
    const networkId = await web3.eth.net.getId();
    // console.log(networkId);
    const manageCID = new web3.eth.Contract([{"inputs":[{"internalType":"string","name":"cid","type":"string"}],"name":"addCID","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"showCID","outputs":[{"internalType":"string[]","name":"","type":"string[]"}],"stateMutability":"view","type":"function"}], "0x1cdD37d897793E85c65b525f5e98B7715656FAd2");
    const cidArray = await manageCID.methods.showCID().call();
    console.log(`New data value: ${cidArray}`);
    console.log(cidArray.length)

    const file_path = "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/TF_Models/client";
    for(let i=1; i<=cidArray.length; i+=1){
        console.log(cidArray[i-1])
        console.log("ipfs cat "+cidArray[i-1]+" > "+file_path+i+".h5")
        childProcess.execSync("ipfs cat "+cidArray[i-1]+" > "+file_path+i+".h5");
        //ipfs cat QmSCsSuKvPbs36cdot7eeUu3ZkH3LpNLTPs9zQSHm1JNLL > C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/TF_Models/client1.h5
    }

}

showCID();
// const data = childProcess.execSync("ipfs cat +"cid[i]"+ > +"file_path"+1+".h5");
// const cid = data.toString();
// console.log("sliced"+cid.slice(6,52))