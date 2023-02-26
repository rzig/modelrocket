import "./App.css";

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
        As easy as command-C
      </Heading>
      <Text fontSize={{ base: "lg" }} marginBottom={10}>
        Simply make an account, then run <tt> modelrocket.upload</tt> after
        you've trained your model. We'll deploy your model on our cluster, and
        make an API for you. No wait. No fuss. And when you're ready to update
        your model, just call <tt> upload</tt> again: we'll take care of the
        rest.
      </Text>
    </Container>
  );
}

function App() {
  // 2. Wrap ChakraProvider at the root of your app
  return (
    <ChakraProvider theme={theme}>
      <WithSubnavigation />
      <CallToActionWithAnnotation />
      <Train />
    </ChakraProvider>
  );
}

export default App;
