const fs = require('fs');
const os = require('os');
const express = require('express');
const mongodb = require('mongodb');
const uuid = require('uuid');
const dotenv = require("dotenv");
const bodyParser = require('body-parser');
const multer  = require('multer');

dotenv.config();

const initObjectStoreClient = require("@relaycorp/object-storage").initObjectStoreClient;
console.log(process.env)
const s3client = initObjectStoreClient("minio", "http://localhost:9000", process.env.ACCESS_KEY_ID, process.env.SECRET_ACCESS_ID, false);

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
        key:uuid.v4(),
        name:n,
        input: {
            type: i.type,
            shape: i.shape
        }
        // shard: {
        //     ids: ['0']
        // }
    };
    return template;
};

const uploadToS3 = (path, uuid) => {
    const f = fs.readFileSync(path);
    return s3client.putObject({body: f}, uuid, "models");
}


server.get('/', async (req, res) => {
    const whatever = await db.collection('models').findOne({ name: "test" })
    res.send(whatever);
});
server.post('/load_model', upload.single('model'), async function (req, res) {
    const collection = await db.collection('models')
    const request = req.body;
    console.log(req.body);
    
    if (req.body == null || req.body.name == null || req.body.input == null || req.body.input.type == null || req.body.input.shape == null) {
        console.log("req contains NULL values, rejecting query");
        return res.status(500);
    }
    const new_record = await generateRecord(request.name, request.input);
    collection.insertOne(new_record);
    return res.json({ key: new_record.key });
});

server.post('/upload_model_file/:model_key', upload.single('model'), async (req, res) => {
    const file = req.file;
    console.log(file);
    console.log(typeof file.path);
    const model_key = req.params.model_key;
    const entry = await db.collection('models').findOne({ key: model_key });
    await uploadToS3(file.path, entry.uuid);
    return res.json({"a": "b"});
});


setup().then(() => {
    server.listen(3000, () => {
        console.log("Listening on the port 3000...");
    });
});