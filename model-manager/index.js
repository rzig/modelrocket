const express = require('express');
const mongodb = require('mongodb');
const uuid = require('uuid');
const dotenv = require("dotenv");
const bodyParser = require('body-parser');
const s3 = require('s3-client');
const server = express();

server.use(bodyParser.json());
server.use(bodyParser.urlencoded({ extended: true })); 

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
            input_type: i.input_type,
            size: i.size
        },
        key: uuid.v1(),
        shard: {
            ids: ['0']
        }
    };
    return template;
};

const s3Client = s3.createClient({
    maxAsyncS3: 20,     // this is the default
    s3RetryCount: 3,    // this is the default
    s3RetryDelay: 1000, // this is the default
    multipartUploadThreshold: 20971520, // this is the default (20 MB)
    multipartUploadSize: 15728640, // this is the default (15 MB)
    s3Options: {
      accessKeyId: process.env.ACCESS_KEY_ID,
      secretAccessKey: process.env.SECRET_ACCESS_ID,
      region: "Best Country",
      endpoint: 'http://127.0.0.1:9001/',
      sslEnabled: false
      // any other options are passed to new AWS.S3()
      // See: http://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/Config.html#constructor-property
    },
  });

setup().then(() => {
    
    server.listen(3000, () => {
        console.log("Listening on the port 3000...");
    });
})
server.get('/', async (req, res) => {
    const whatever = await db.collection('models').findOne({name:"test"})
    res.send(whatever);
});
server.post('/load_model', async function (req, res) {
    const collection = await db.collection('models')
    const request = req.body;
    //console.log(req.body);
    console.log(req.body);
    if (req.body.name == null || req.body.input == null || req.body.input.input_type == null || req.body.input.size == null) {
        console.log("req contains NULL values");
    }
    const new_record = await generateRecord(request.name, request.input);
    key = new_record.key;
    //console.log(new_record);
    //collection.insertOne(new_record);
    return res.json({"key": key});
});




