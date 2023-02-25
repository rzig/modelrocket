const express = require('express');
const mongodb = require('mongodb');
const uuid = require('uuid');
const dotenv = require("dotenv");

const server = express();
dotenv.config();
let client = null;
let db = null;

const setup = async () => {
    client = await mongodb.MongoClient.connect(process.env.MONGODB_URL);
    db = client.db('Models');
}

const generateRecord = async (n, i) => {
    const template = {
        uuid:uuid.v4(),
        name:n,
        input: {
            type: i.type,
            size: i.size
        },
        key: uuid.v1(),
        shard: {
            ids: ['0']
        }
    };
    return template;
};

setup().then(() => {
    post();
    server.listen(3000, () => {
        console.log("Listening on the port 3000...");
    });
})

server.get('/', async (req, res) => {
    const whatever = await db.collection('models').findOne({name:"test"})
    res.send(whatever);
});

server.post('/', async (req, res) => {
    const collection = await db.collection('models')
    const request = req.json();
    if (request.name == null || request.input == null || request.input.type == null || request.input.size == null) {
        console.log("req contains NULL values");
    }
    const new_record = generateRecord(request.name, request.input);
    collection.insertOne(new_record);
});



