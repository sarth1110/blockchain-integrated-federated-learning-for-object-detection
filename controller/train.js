const express = require('express');
const router = express.Router();
const Request = require('request');
const childProcess = require('child_process');
const contract = require('../services/contract');


router.post("/trainModel", (request, response)=> {
    var start = new Date();
    const options = {
        url: 'http://127.0.0.1:5000/trainModel',
        json: true,
        body: {
            "clientNo" : request.body.clientNo
        }
    };

    let request_call = new Promise((resolve, reject) => {
        Request.post(options, async (err, res, body) => {
            if (err) {
                reject(err);
                return console.log(err);
            }

            resolve(body.toString());
            console.log(`Status: ${res.statusCode}`);
            response.send("Model Trained Successfully");
        });
    });

    request_call.then((response) => {
        console.log(response);
        //add model to ipfs
        const model = childProcess.execSync("ipfs add "+response)
        const _cid = model.toString().slice(6,52);
        console.log(_cid);
        //Adding CID to the blockchain
        contract.addLocalCID(_cid);
        var end = new Date() - start;
        console.log(end);
    }).catch((error) => {
        console.log(error);
    });
    
});

router.post("/testModel", (request, response)=>{
    var start = new Date();
    const gcid = contract.showGlobalCID();
    gcid.then(async (res)=>{
        console.log(res)
        childProcess.execSync("ipfs cat "+res+" > C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Aggregated_Model_for_testing/agg_model.h5");
        
        const options = {
            url: 'http://127.0.0.1:5000/testModel',
            json: true,
            body: {
                    "model_path" : "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Aggregated_Model_for_testing/agg_model.h5",
                    "test_images" : "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Images/client_1/test"
            }
        };
    
        let request_call = new Promise((resolve, reject) => {
            Request.post(options, async (err, res, body) => {
                if (err) {
                    reject(err);
                    return console.log(err);
                }
    
                resolve(body.toString());
                console.log(`Status: ${res.statusCode}`);
                response.send("Testing Done!!")
            });
        });
    
        request_call.then((response) => {
            console.log("Accuracy: "+response);
            var end = new Date() - start;
            console.log(end);
        }).catch((error) => {
            console.log(error);
        });
    })
    .catch((err)=>{ console.log(err); });

})

module.exports = router;