const fs = require("fs");
const os = require("os");
const express = require("express");
const mongodb = require("mongodb");
const uuid = require("uuid");
const dotenv = require("dotenv");
const bodyParser = require("body-parser");
const multer = require("multer");
var crypto = require("crypto");
const fetch = require("node-fetch");
const cors = require("cors");
dotenv.config();

const initObjectStoreClient =
  require("@relaycorp/object-storage").initObjectStoreClient;
const s3client = initObjectStoreClient(
  "minio",
  "http://127.0.0.1:9000",
  process.env.ACCESS_KEY_ID,
  process.env.SECRET_ACCESS_ID,
  false
);

const upload = multer({ dest: os.tmpdir() });

const server = express();
server.use(cors());
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({ extended: true }));

let client = null;
let db = null;

const setup = async () => {
  client = await mongodb.MongoClient.connect(process.env.MONGODB_URL);
  db = client.db("Models");
};

const generateRecord = async (n, i) => {
  const template = {
    uuid: uuid.v4(),
    key: uuid.v4(),
    name: n,
    input: {
      type: i.type,
      shape: i.shape,
    },
    new:true
    // shard: {
    //     ids: ['0']
    // }
  };
  return template;
};

const uploadToS3 = (path, uuid) => {
  const f = fs.readFileSync(path);
  return s3client.putObject({ body: f }, uuid, "models");
};

server.get("/get_models/", async (req, res) => {
  try {
    const allRecords = await db.collection("models").find({}).toArray();
    res.send(allRecords);
  } catch (e) {
    res.status(500);
    return;
  }
});
server.get("/get_model/:token", async (req, res) => {
  try {
    const token = req.params.token;
    const record = await db.collection("models").findOne({ key: token });
    res.send(record);
  } catch (e) {
    res.status(500);
    return;
  }
});

server.get("/get_new_token/:uuid", async (req, res) => {
  const new_uuid = uuid.v4();
  const new_token = crypto
    .createHash("sha256")
    .update(new_uuid)
    .digest("base64");
  try {
    const provided_key = req.params.uuid;
    const record = await db
      .collection("models")
      .updateOne({ uuid: provided_key }, { $set: { key: new_token } });
    res.send({ token: new_uuid });
  } catch (e) {
    res.status(500);
    return;
  }
});

server.post("/load_model", upload.single("model"), async function (req, res) {
  const collection = await db.collection("models");
  const request = req.body;
  // console.log(req.body);

  if (
    req.body == null ||
    req.body.name == null ||
    req.body.input == null ||
    req.body.input.type == null ||
    req.body.input.shape == null
  ) {
    console.log("req contains NULL values, rejecting query");
    return res.status(500);
  }
  const exists = collection.find({ name: req.body.name }).count() > 0;
  if (exists) {
    const record = collection.findOne({ name: req.body.key });
    collection.updateOne(
      { name: req.body.name },
      {
        $set: {
          input: { type: req.body.input.type, shape: req.body.input.shape },
        },
      }
    );
    return res.json({ token: record.key, key: record.uuid });
  }
  const new_record = await generateRecord(request.name, request.input);
  const key = new_record.key;
  new_record.key = crypto.createHash("sha256").update(key).digest("base64");
  collection.insertOne(new_record);
  await fetch(
    `http://127.0.0.1:5005/register_hash_to_model?model=${encodeURIComponent(
      new_record.uuid
      )}&hash=${encodeURIComponent(new_record.key)}`
      );
      return res.json({ token: new_record.key, key: new_record.uuid });
    });
    
    server.post(
      "/upload_model_file/:model_uuid",
      upload.single("model"),
      async (req, res) => {
        const file = req.file;
        // console.log(file);
        // console.log(typeof file.path);
        const model_uuid = req.params.model_uuid;
        const collection = await db.collection("models");
        const entry = collection.findOne({ uuid: model_uuid });
        await uploadToS3(file.path, entry.uuid);
        await fetch(
          `http://127.0.0.1:5005/update_model?model=${encodeURIComponent(model_uuid)}`
        );
        if (entry.new) {
          collection.updateOne({ uuid: model_uuid },{ $set:{ new: false } });
          await fetch(
          `http://127.0.0.1:5005/generate_shard?model=${encodeURIComponent(model_uuid)}`
          );
        }
    return res.json({ status: "success" });
  }
);

setup().then(() => {
  server.listen(3000, () => {
    console.log("Listening on the port 3000...");
  });
});
