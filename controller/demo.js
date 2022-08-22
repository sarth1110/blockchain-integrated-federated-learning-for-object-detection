const express = require("express");
const router = express.Router();
const request = require('request');

router.post("/", (req, res)=> {
    console.log(req.body);
    const options = {
        url: 'http://127.0.0.1:5000/data',
        json: true,
        body: {
            "clientNo" : req.body.clientNo
        }
    };

    let data = request.post(options, (err, res, body) => {
        if (err) {
            return console.log(err);
        }
        console.log(`Status: ${res.statusCode}`);
        console.log(body);    
    });
    //console.log(data);
    
});

module.exports = router;