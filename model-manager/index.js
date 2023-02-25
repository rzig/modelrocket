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

const template = {
    uuid:"abcdef",
    name:"test",
    input: {
        type: "int",
        size: 3
    },
    key: 4372,
    shard: {
        id: 0
    }
};

const post = async () => {
    const collection = await db.collection('models')
    const new_record = template;
    new_record.uuid = uuid.v4();
    collection.insertOne(template);
};