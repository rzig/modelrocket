import "./App.css";
import { Link, Route } from "wouter";

import {
  Card,
  CardBody,
  CardHeader,
  ChakraProvider,
  Grid,
  GridItem,
  ListItem,
  UnorderedList,
} from "@chakra-ui/react";
import { useState } from "react";

import SyntaxHighlighter from "react-syntax-highlighter";
import {
  docco,
  darcula,
  nord,
} from "react-syntax-highlighter/dist/esm/styles/hljs";
import {
  dark,
  atomDark,
  duotoneDark,
} from "react-syntax-highlighter/dist/esm/styles/prism";

import {
  Box,
  Heading,
  Container,
  Text,
  Button,
  Stack,
  extendTheme,
} from "@chakra-ui/react";
import WithSubnavigation from "./components/Navbar";

import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
} from "@chakra-ui/react";

import { ArrowRightIcon } from "@chakra-ui/icons";
import { useEffect } from "react";
import SmallWithNavigation from "./components/Footer";

const theme = extendTheme({
  fonts: {
    heading: `'Inter'`,
    body: `'Inter'`,
  },
});

function CallToActionWithAnnotation() {
  return (
    <>
      <Container maxW={"3xl"}>
        <Stack
          as={Box}
          textAlign={"center"}
          spacing={{ base: 8, md: 14 }}
          py={{ base: 20, md: 36 }}
        >
          <Heading
            fontWeight={600}
            fontSize={{ base: "3xl", sm: "5xl", md: "7xl" }}
            lineHeight={"110%"}
          >
            Deploy any model <br />
            <Text as={"span"} color={"green.400"}>
              in one line.
            </Text>
          </Heading>
          <Text color={"gray.500"} fontSize={{ base: "xl" }}>
            With ModelRocket, the sky's the limit. Deploy a model from any
            framework to our serverless platform to access inference
            immediately. We'll take care of security, scaling, and performance,
            so that you can focus on development.
          </Text>
          <Stack
            direction={"row"}
            spacing={3}
            align={"center"}
            alignSelf={"center"}
            position={"relative"}
          >
            <Button
              colorScheme={"white"}
              bg={"white"}
              _hover={{
                bg: "white",
              }}
              fontSize={{ base: "xl" }}
              width={200}
              height={50}
              color="gray.500"
              borderColor={"gray.500"}
              borderWidth={2}
            >
              Learn More
            </Button>
            <Button
              colorScheme={"green"}
              bg={"green.400"}
              _hover={{
                bg: "green.500",
              }}
              fontSize={{ base: "xl" }}
              width={200}
              height={50}
            >
              Get Started
            </Button>
          </Stack>
        </Stack>
      </Container>
    </>
  );
}

const vj_code = `import torch
import torch.nn as nn
import torch.optim as optim
import modelrocket

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Define your model just like normal


def train(args, model, device, train_loader, optimizer, epoch):
    # Train as you normally would, and upload when you're done!
    modelrocket.upload("digit recognizer", model, {"digits": ...})
    # Model URL printed to terminal and returned
`;

const vj_code_2 = `fetch("http://127.0.0.1:5000/inference", {
  method: "POST",
  body: JSON.stringify({
    "token": "YOUR_TOKEN_HERE", 
    "model": "YOUR_MODEL_HERE",
    "inputs": {MODEL_INPUTS_HERE}
  }),
  headers: {
      "Content-type": "application/json; charset=UTF-8"
  }
})
.then((response) => response.json())
.then((json) => {
  console.log(json.result); // Your model's output is here!
}`;

function Train() {
  return (
    <Container maxW={"4xl"}>
      <Box backgroundColor={"gray.800"} borderRadius={10} color="white">
        <SyntaxHighlighter
          language="python"
          style={{ ...nord, borderRadius: "10px" }}
        >
          {vj_code}
        </SyntaxHighlighter>
      </Box>

      <Heading as="h3" marginTop={6} textAlign="center" marginBottom={"2.5"}>
        As easy as control-C
      </Heading>
      <Text fontSize={{ base: "lg" }} marginBottom={10}>
        Simply make an account, then run <tt> modelrocket.upload</tt> after
        you've trained your model. We'll deploy your model on our cluster, and
        make an API for you. No wait. No fuss. And when you're ready to update
        your model, just call <tt> upload</tt> again: we'll take care of the
        rest.
      </Text>

      <Box backgroundColor={"gray.800"} borderRadius={10} color="white">
        <SyntaxHighlighter
          language="javascript"
          style={{ ...nord, borderRadius: "10px" }}
        >
          {vj_code_2}
        </SyntaxHighlighter>
      </Box>
      <Heading as="h3" marginTop={6} textAlign="center" marginBottom={"2.5"}>
        ...and control-V
      </Heading>
      <Text fontSize={{ base: "lg" }} marginBottom={10}>
        Once a model is deployed, you can access it from your favorite
        programming language&mdash;just send an HTTP request. As your model
        receives requests, we'll automatically scale it across our cluster to
        ensure that your users never have a bad experience.
      </Text>
    </Container>
  );
}

function Home() {
  return (
    <>
      <WithSubnavigation />
      <CallToActionWithAnnotation />
      <Train />
      <SmallWithNavigation />
    </>
  );
}

const MODELS = [
  {
    uuid: "adsfasdf",
    name: "Cool Model 1",
  },
  {
    uuid: "adsfasdf",
    name: "Cool Model 2",
  },
  {
    uuid: "adsfasdf",
    name: "Cool Model 3",
  },
];

function ModelComponent({ model }) {
  const codeSample = `fetch("http://127.0.0.1:5000/inference", {
  method: "POST",
  body: JSON.stringify({
    "token": "YOUR_TOKEN_HERE", 
    "model": "${model.uuid}",
    "inputs": {MODEL_INPUTS_HERE}
  }),
  headers: {
      "Content-type": "application/json; charset=UTF-8"
  }
})
.then((response) => response.json())
.then((json) => {
  console.log(json.result); // Your model's output is here!
}`;
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [tokenGenerated, setTokenGenerated] = useState(false);
  const [token, setToken] = useState("");

  const getToken = () => {
    fetch(`http://localhost:3000/get_new_token/${model.uuid}`)
      .then((r) => r.json())
      .then((t) => {
        setToken(t.token);
        setTokenGenerated(true);
      });
  };

  return (
    <>
      <Box
        borderWidth={2}
        borderRadius={7}
        borderColor="gray.300"
        px={5}
        py={5}
        _hover={{
          borderColor: "gray.500",
          cursor: "pointer",
        }}
        marginBottom={2}
        onClick={onOpen}
      >
        <Stack
          direction={"row"}
          align={"center"}
          alignSelf={"center"}
          position={"relative"}
          justifyContent={"space-between"}
        >
          <Text fontSize={{ base: "lg" }}>{model.name}</Text>
          <ArrowRightIcon color="gray:600" fill="gray:500" opacity={0.85} />
        </Stack>
      </Box>

      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent maxW="750px">
          <ModalHeader>{model.name}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Heading fontSize={{ base: "lg" }}>Access Your Model</Heading>
            <SyntaxHighlighter
              language="javascript"
              style={{ ...nord, borderRadius: "10px" }}
            >
              {codeSample}
            </SyntaxHighlighter>
            <Heading fontSize={{ base: "lg" }} marginTop={4}>
              Need a new key?
            </Heading>
            {!tokenGenerated && (
              <Button
                as={"a"}
                display={{ base: "none", md: "inline-flex" }}
                fontSize={"sm"}
                fontWeight={600}
                color={"white"}
                bg={"green.400"}
                href={"#"}
                _hover={{
                  bg: "green.500",
                }}
                onClick={getToken}
              >
                Regenerate Token
              </Button>
            )}
            {tokenGenerated && <tt style={{ fontSize: 15 }}>{token}</tt>}
          </ModalBody>
          <ModalFooter></ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}

function Admin() {
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState([]);
  useEffect(() => {
    fetch("http://localhost:3000/get_models/")
      .then((res) => res.json())
      .then((j) => {
        setModels(j);
        setLoading(false);
      });
  }, []);
  return (
    <Container maxW={"3xl"}>
      <Heading as="h1" marginTop={4} marginBottom={3}>
        My Models
      </Heading>
      {!loading &&
        models.map((model) => {
          return <ModelComponent model={model} />;
        })}
      {loading && <Text textAlign="center">Loading...</Text>}
    </Container>
  );
}

function App() {
  // 2. Wrap ChakraProvider at the root of your app
  return (
    <ChakraProvider theme={theme}>
      <Route path="/">
        <Home />
      </Route>
      <Route path="/admin">
        <Admin />
      </Route>
    </ChakraProvider>
  );
}

export default App;
