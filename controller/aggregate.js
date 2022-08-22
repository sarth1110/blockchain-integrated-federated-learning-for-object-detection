const express = require("express");
const router = express.Router();
const Request = require('request');
const contract = require('../services/contract');
const childProcess = require('child_process');

router.post("/aggregateModel", async (request, response)=> {
    var start = new Date();
    const file_path = "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/TF_Models/client";
    const cidArray = contract.showLocalCID();
    cidArray.then((result)=>{
        console.log(result);
        for(let i=1; i<=result.length; i+=1){
                
            childProcess.execSync("ipfs cat "+result[i-1]+" > "+file_path+i+".h5");
            //test downloaded model(pass threshold)
            const optionsT = {
                url: 'http://127.0.0.1:5000/testModel',
                json: true,
                body: {
                    "model_path" : file_path+i+".h5",
                    "test_images" : "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Images/client_1/test"
                }
            };
        
            let request_call = new Promise((resolve, reject) => {
                Request.post(optionsT, async (err, res, body) => {
                    if (err) {
                        reject(err);
                        return console.log(err);
                    }
        
                    resolve(body.toString());
                    console.log(`StatusV: ${res.statusCode}`);
                });
            });
        
            request_call.then((response) => {
                console.log("Accuracy: "+response);
                if(response<80){
                    childProcess.execSync("rm "+file_path+i+".h5");
                }
            }).catch((error) => {
                console.log(error);
            });
        }
        
        const optionsA = {
            url: 'http://127.0.0.1:5000/aggregateModel',
            json: true,
            body: {
                "model_directory" : "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models"
            }
        };
        
        let request_call = new Promise((resolve, reject) => {
            Request.post(optionsA, (err, res, body) => {
                if (err) {
                    reject(err);
                    return console.log(err);
                }
                console.log(`StatusA: ${res.statusCode}`);
                resolve(body.toString());   
                response.send("Model Aggregated Successfully");
            })
        });
    
        request_call.then((response) => {
            console.log(response);
    
            //add model to ipfs
            const model = childProcess.execSync("ipfs add "+response)
            const _cid = model.toString().slice(6,52);
    
            //Adding CID to the blockchain
            contract.addGlobalCID(_cid);
            var end = new Date() - start;
            console.log(end);
        }).catch((error) => {
            console.log(error);
        });
    })
    .catch((error)=>{
        console.log(error);
    })

});

module.exports = router;