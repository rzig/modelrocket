const express = require('express');
const mongodb = require('mongodb');
const uuid = require('uuid');
const dotenv = require("dotenv");
const bodyParser = require('body-parser');
const os = require('os');
const s3 = require('s3-client');
const multer  = require('multer');

dotenv.config();
const upload = multer({ dest: os.tmpdir() });

const server = express();
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({ extended: true })); 

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
            shape: i.shape
        },
        key: uuid.v4(),
        shard: {
            ids: ['0']
        }
    };
    return template;
};

const s3Client = s3.createClient({
    maxAsyncS3: 20,
    s3RetryCount: 3,
    s3RetryDelay: 1000,
    multipartUploadThreshold: 20971520, // 20 MB
    multipartUploadSize: 15728640,      // 15 MB
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
    const whatever = await db.collection('models').findOne({ name: "test" })
    res.send(whatever);
});
server.post('/load_model', upload.single('model'), async function (req, res) {
    const collection = await db.collection('models')
    const request = await req.body;
    const file = req.file;
    console.log(file);
    console.log(req.body);
    if (req.body == null || req.body.name == null || req.body.input == null || req.body.input.type == null || req.body.input.shape == null) {
        console.log("req contains NULL values, rejecting query");
        return res.status(500);
    }
    const new_record = await generateRecord(request.name, request.input);
    //console.log(new_record);
    //collection.insertOne(new_record);
    return res.json({ key: new_record.key });
});




