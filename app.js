const express = require('express');
const app = express();
const bodyParser = require("body-parser");

const TrainModel = require('./controller/train');
const AggregateModel = require('./controller/aggregate');
const Demo = require('./controller/demo');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended : true
}));

app.use("/local", TrainModel);
app.use("/server", AggregateModel);
app.use("/demo", Demo);

app.listen(3000, ()=>{
    console.log("Server listening");
})