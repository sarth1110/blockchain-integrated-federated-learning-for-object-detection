const Web3 = require('web3');

const address = "0x5B245bAdA3E41692F507e8a9eEf3107733a97BC0";
const privateKey = "2a25255ce1801666fd7c549cfb8e63dc54abab2d576f3b01cd653bc3060740cf";
var web3;
var networkId;
var manageCID;
var gasPrice;
var nonce;

const setETHdetails = async()=>{
    web3 = new Web3(new Web3.providers.HttpProvider("https://ropsten.infura.io/v3/0a55ff1dc9ff4d26999876d698f0b748"));
    networkId = await web3.eth.net.getId();
    manageCID = new web3.eth.Contract([{"inputs":[{"internalType":"string","name":"_gcid","type":"string"}],"name":"addGlobalCID","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"_lcid","type":"string"}],"name":"addLocalCID","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"showGCID","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"showLCID","outputs":[{"internalType":"string[]","name":"","type":"string[]"}],"stateMutability":"view","type":"function"}], "0xb1CEa642E7E7AE20099ecc452B785F084462992E");

    gasPrice = await web3.eth.getGasPrice();
    const accountNonce = '0x' + (web3.eth.getTransactionCount(address) + 1).toString(16)
}

const addLocalCID = async(_cid)=>{
    const tx = manageCID.methods.addLocalCID(_cid);
    const gas = await tx.estimateGas({from: address});
    const data = tx.encodeABI();
    const signedTx = await web3.eth.accounts.signTransaction(
    {
        to: manageCID._address, 
        data,
        gas,
        gasPrice,
        nonce, 
        chainId: networkId
    },
        privateKey
    );
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log(`Transaction hash: ${receipt.transactionHash}`);
}


const addGlobalCID = async(_cid)=>{
    const tx = manageCID.methods.addGlobalCID(_cid);
    const gas = await tx.estimateGas({from: address});
    const data = tx.encodeABI();
    const signedTx = await web3.eth.accounts.signTransaction(
    {
        to: manageCID._address, 
        data,
        gas,
        gasPrice,
        nonce, 
        chainId: networkId
    },
        privateKey
    );
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log(`Transaction hash: ${receipt.transactionHash}`);
}
    
const showLocalCID = async()=>{
    const cidArray = await manageCID.methods.showLCID().call();
    return cidArray;
}

const showGlobalCID = async()=>{
    const cidArray = await manageCID.methods.showGCID().call();
    return cidArray;
}

setETHdetails();

module.exports = {
    addGlobalCID,
    addLocalCID,
    showGlobalCID,
    showLocalCID
}