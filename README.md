CrossCracker 2 is an AI-assisted crossword solving application that combines large language models, retrieval-augmented clue hints from a Chroma vector database, and a step-by-step heuristic solver using a sophisticated constraint satisfaction algorithm to fill crossword grids interactively.  It includes a FastAPI backend, a browser-based frontend, support for local Ollama models or Anthropic Claude, and tooling for recording, replaying, and inspecting solve sessions so you can study how the solver reasons through placements, backtracking, and fallbacks.

Depending on the selected LLM, CrossCracker 2 has proved itself capable of solving up to a Tuesday New York Times crossword puzzle with no errors in about 10 minutes.  It can solve most clues for puzzles beyond that level too, but such puzzles typically also include "tricks" like rebuses or self-referential clues that my solver is not built to address.

APP ARCHITECTURE
----------------
**1. The Solver Algorithm**

Believe it or not, all the AI stuff described below was the easy part.  I handled all that in just a few days, and most of THAT was faffing around with the LLM prompt.

The really hard part was the solver algorithm itself -- a good old-fashioned coding problem.  I'm using a variant of a **constraint satisfaction algorithm** with **dependency-aware backtracking**.  What this means is that the app can guess an answer, and if it turns out to be wrong it can backtrack and guess again, without getting stuck, going into an infinite loop, or taking longer than the lifetime of the universe to solve a puzzle.

The main difference from the first version of the crossword solver is that version 2 guesses the answers for each clue at runtime.  Technically, Version 1 of this app did use an LLM to get the answers, but I ran the LLM and recorded its answers *before* running the solver app, and made sure the *correct* answers were in the list.  That meant there was a small, fixed set of possible answers for each clue, guaranteed to include the correct answer, when I ran the solver.  Therefore it could solve the puzzle using a simple depth-first search.

Letting the LLM generate guesses in real time while the app runs, and thus having an effectively infinite set of possible answers per clue -- not necessarily including the correct answer -- is a literally exponentially more difficult problem.  Version 2 had to be VASTLY better at guessing clues, even though it's still using the same LLM as version 1.  Thanks to RAG technology, it is.  It usually gets over 90% of the clues correct on the first guess.

Still, it's not 100%, so the solving algorithm also has to be much smarter.  Smarter about how to choose the order in which to answer clues, how to generate candidate answers and which ones to select, and most importantly, how and when to "backtrack" and try something different when it gets stuck.  Frankly, the AIs I consulted -- and I asked many of the big ones -- were absolutely hopeless at this task.  Even with a lot of hand-holding, the best AIs out there are still completely incapable of properly solving a problem of this complexity.  When asked, they will confidently give you... a steaming pile of dung.  This confirmed my conviction that even top LLMs are good only at small, focused tasks.  If you want high-level design of any quality, you still need a human.

I chose to model the algorithm after how humans (like me!) solve crosswords IRL.  The algorithm uses a complex set of heuristics to determine which clue to fill in first, based on parameters including candidate answer confidence, the length of the answer, and the number of already filled-in letters.  When an answer is placed, the LLM generates new candidate answers for each clue that had a new letter filled in.  It proceeds in this manner until it solves the puzzle or gets stuck.  In that case, it uses another complex set of heuristics to guess which clue to "backtrack".  It then applies a penalty to the clue it removed, but doesn't remove it from the list of potential candidate answers entirely.

Sometimes, though, even with just one letter left to fill in a given clue, an LLM can't guess the correct answer.  Crossword clues are famously punny and LLMs are generally not trained on that kind of stuff, so even an answer that seems completely obvious to a human just isn't in the LLMs vocabulary.  In that case my algorithm will thrash a few times, and then finally give up and use a "fallback" -- it fills in the clue's known correct answer from a given list (i.e., it cheats).

My goal in coding the solver is to always reduce fallbacks to 0.  With the LLMs I've used, it can do that most of the time on a NYT Tuesday puzzle.  Once in a while it can solve a Wednesday puzzle without resorting to a fallback, but usually it requires several of them.  And starting on Thursdays the NYT puzzles are filled with rebuses (grid squares that contain more than one letter or sometimes even a symbol) and similar gimmicks that my solver is not equipped to handle, so that's basically as far as it can go.

**2. ChromaDB and RAG**

The app uses a ChromaDB vector database to store a knowledge base of past crossword clue-answer pairs.  It was populated using a dataset obtained from Hugging Face with about 6.4 million rows.

Vector databases use a neural network to retrieve data "similar" to the given input.  This makes them useful for all sorts of things, from helpbots to medical diagnosis apps.  In this app, we use the vector database on startup to get a set of clue-answer pairs that are "semantically similar" to each clue in the puzzle.  In other words, we get a list of "hints" for each clue.  We then pass those hints to an LLM to help it select and rank a list of possible candidate answers for a clue.  This is classic RAG technology -- Retrieval Augmented Generation.

IMPORTANT: ChromaDB is a memory hog.  To do anything, it has to load the entire database index into memory.  That's just how vector DBs work.  In the case of our 6.4 million row crossword database, that index consumes about 11.5 GB of memory.  Yikes!!!  It takes a long time (about 50 seconds on my system) to load the index the first time you run a query, and it unloads automatically after the keep-alive expires.  The default keep-alive was something like 30 seconds, but I've overridden it in the code to 10 minutes.

NOT IMPORTANT BUT INTERESTING: With the code using Chroma's preferred ONNXMiniLM_L6_V2 "embedding function" (which defines the mathematical procedure used to access the database), the call stack looks like this:

**app -> chromaDB library -> ONNX Runtime -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware**

But if it uses the SentenceTransformerEmbeddingFunction embedding function (which is said to be slower and more memory intensive, but has additional features), the call stack would be:

**app -> chromaDB library -> PyTorch -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware**

Hey, there's PyTorch!  Everyone loves PyTorch because it helped usher in the current age of AI programming.  It's a hot name when you're looking at job requirements.  But we're not using it.  It's slow and cumbersome and unnecessary.

ALSO INTERSTING: While ChromaDB's CUDAExecutionProvider has vastly superior performance to the CPUExecutionProvider (as you would expect, since it uses the GPU instead of the CPU), it doesn't really make any difference for our little crossword app.  For all the massive memory requirements ChromaDB imposes on us, in its entire run the solver executes just ONE database query which takes only a second or two, regardless of whether you're using the GPU or CPU.  The overwhelming majority of the time required to use Chroma is when it initially loads the database index.


**3. Ollama and LLM support**

When we want to solve a particular crossword entry, we collect the clue, the "hints" we got from the vector database, and any other contextual information like the answer's length and the current pattern of known letters.  We pass all this data to an LLM (either a local LLM managed by Ollama or an Anthropic LLM in the cloud) along with a carefully constructed prompt instructing it to guess a few answers.  Then we call the LLM again and ask it to score these answers with a confidence level from 0-100.

Why two separate calls, you ask?  Because LLMs are dumb.  If you ask them to rank their own answers in the first call, they'll rank everything at or near 100%.  Remember, the LLM chose those answers in the first place because they satisfied its own internal criteria, so in the LLM's opinion they're all good.  Asking the LLM to rank answers in a separate call forces it to consider them more carefully and you get much better results -- though it still consistently overrates its confidence levels.

Therefore we also use the "logprobs" obtained in the first query.  Logprobs ("log probabilities") are the internal probability the LLM used to predict each particular token in the response, so they're also a sort of "confidence" in the answer.  But they're just mechanical measures, they don't consider the semantics of the crossword clue.  Usually the logprobs and the confidence levels reported in the text response from the second query are correlated, but sometimes one is way off.  Therefore we use a weighted average of the two values to calculate the effective confidence in each crossword answer.  That's almost always good enough.

Speaking of internal API values, we also use the "popular" LLM tuning parameters when querying it for candidate answers: temperature (which is how "speculative" or "aggressive" the LLM is in selecting tokens), top_p (which limits the probability of the tokens it selects), and top_k (which limits the number of tokens selected).  They're really just parameters for the shape of the "Receiver Operating Characteristic" (ROC) curve, which is the technical name of the curve in a machine learning classification calculation that separates the positive and negative cases.

[UPDATE: Anthropic has deprecated use of these parameters in their newer models because they feel it unduly exposes the innards of the model's execution to the outside.]

We use three sets of these parameters, depending on the "search level" for a particular clue.  When we're stuck on a clue and don't have any candidate answers for it, we increase its search level.  This gives us more candidates.  Most of them are usually garbage -- not even real words -- but occasionally it gives us something useful.  (Though I want to run some metrics to see how accurate that statement is!)

Currently we're using the llama3.1:8b LLM, which basically has the intelligence of Elmer Fudd when compared to the LLMs you use online every day, but that's about all I can manage with the horsepower at my disposal on my potato PC.  For comparison, there's a llama3.1:405b model available on Ollama.  That's *50 times* as powerful (and 50 times the memory requirements) as the model I'm using.  Sheesh!  Maybe one day I'll set up my app so it can call a powerful online LLM instead -- but that costs money, yo!

IMPORTANT: The LLM we're using needs 6-8 GB of RAM.  Combined with the 12 GB or so required by the vector database, that puts a pretty heavy load on WSL's memory, which I have capped at 20 GB.  Expect a lot of swapping and thrashing.

**4. FastAPI**

This app uses a FastAPI framework for the "back end".  FastAPI is basically a lightweight REST API framework similar to Flask, but it uses ASGI instead of WSGI (read: it's faster and more modern).

But note that CrossCracker is not a traditional client-server app that can run across the Internet.  I couldn't put the back end on an app server and then run the interface on my phone.  Everything runs locally.  We're using FastAPI mainly to cleanly separate the front end from the back end and make it easy to build a UI using well-known tools like HTML and JavaScript.  Making it Internet-capable is *possible* (I did it with CrossCracker version 1)... but it's impractical.  It might be intersting to implement that at some point, but we'll cross that bridge if we ever come to it.


INSTALLING THE APP
------------------
NOTE: All of these instructions are executed inside WSL (the Windows Subsystem for Linux) on my PC, which is Windows's Linux emulator.  I've never tested it on a native Linux system, but it should run fine there too.

**1. Get the project off GitHub**
   
If you're reading this, you've already found it!

**2. Create a Python virtual environment**

There are many ways to do this, but if you're using VS Code the easiest way is to use its "Python: Create Environment..." command from the Command Palette (Ctrl+Shift+P).  Be sure to call the directory .venv and put it in {project root}, NOT the /backend subdirectory.

**3. Activate the virtual environment**

If you're running VS Code you usually only have to do this once and then it remembers it for that project.  From the project root, run this in a VS Code Terminal window:

`source ./.venv/bin/activate`

**4. Install app dependencies (FastAPI, uvicorn, etc.)**

Make sure you've activated the virtual environment, then go to {project root}/backend and run:

`pip install -r requirements.txt`

IMPORTANT: One of the project dependencies, ChromaDB, has sub-dependencies on both onnxruntime and onnxruntime-gpu.  BUT THEY ARE MUTUALLY EXCLUSIVE AND CONFLICT WITH EACH OTHER.  Not only that, but BOTH LIBRARIES IT INSTALLS ARE OBSOLETE!  We only want the latest available version of onnxruntime-gpu.  There are several workarounds to this, but the simplest is to run the following after installing the requirements.txt:

```
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

**5. Install ChromaDB dependencies**

To use the selected vector database (ChromaDB) with a GPU, you also need to install *both* the NVidia CUDA Toolkit *and* the NVidia CUDA Deep Neural Network Library (cuDNN), which is built on top of the CUDA Tooklit.  These are not Python libraries, they are separate executables.  They are available here:

```
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn
```

Be sure to get the right versions of everything!  At the time of this writing, we want CUDA Toolkit 12.x and cuDNN 9.x.

**6. Populate the RAG database**

I haven't uploaded the database file to GitHub because it's way too big, but you can create and populate the database from scratch from the HuggingFace dataset I used by running:

`python {project root}/backend/scripts/load_dataset_1.py`

The database will be created in the {project root}/db directory.  The script will download the dataset from the HF website and jam it into the database.  It's free, you just need an access token that you get from the website.  Put the token in an .env file in {project root}:

`HF_TOKEN=mykey`
    
You might have to adjust the path to the .env file because the script is in a subdirectory, I'm not sure.  Anyway, a full load took about *6 hours* on my system, so... get some coffee and put your feet up!

NOTE: You can omit installing the Chroma database and the associated NVIDIA drivers if you don't feel like committing the time and resources to it.  The app will still run, but if you're using a less-capable LLM (i.e., anything you might run via Ollama, or even the lower-tier Claude LLMs), the app will not perform well.  That is, it will fail to solve a lot of clues correctly.


RUNNING THE APP
---------------
**1. Start ChromaDB**

The app expects Chroma to be running in a separate process on local port 8001.  For convenience, I created a "chroma-up" script to start ChromaDB and put it in my ~/.local/bin directory (don't forget to make it executable).  Here are its contents:

```
CHROMA_EXE_DIR=~/dev/crosscracker_2/.venv/bin
CHROMA_DB_DIR=~/dev/crosscracker_2/backend/db
echo "Starting ChromaDB server from $CHROMA_DB_DIR..."
$CHROMA_EXE_DIR/chroma run --path "$CHROMA_DB_DIR" --port 8001
```

I also made a "chroma-down" script to close ChromaDB when you're done for the day.  I'm not quite sure using the "kill" command is a "graceful" way to terminate a process, but it's probably better than closing the terminal window Chroma is running in, or pressing Ctrl+/.  Here are the contents of that script:

```
#!/bin/bash
PID=$(lsof -t -i:8001)
if [ -z "$PID" ]; then
   echo "No Chroma server found on port 8001."
else
   echo "Gracefully stopping Chroma (PID $PID)..."
   kill $PID
fi
```

Note that the crossword database requires about 12 GB of RAM for all of its indexes.  Chroma allocates the memory the first time you run a query against the database (i.e., not when you start Chroma), and keeps it allocated until you stop Chroma.

**2. Start the LLM (optional)**

I'm using Ollama with the llama3.1:8b LLM, which runs in 6-8 GB of RAM.  Feel free to use a bigger/better LLM if you want.  There's a constant in llm.py where you set the model name.  It is NOT NECESSARY to run Ollama manually -- Ollama has a background daemon that services requests and loads the appropriate LLM as necessary.  But you can run the LLM from the command line if you want to query it manually using:

`ollama run llama3.1:8b`

**3. Set up the .env file**

Copy .env.example to .env in the project root:

`cp .env.example .env`

The values in the .env should work fine, except three values corresponding to keys:

   NYT_S_COOKIE

**4. Start the app**

Run this *from the /backend subdirectory*:

`../.venv/bin/uvicorn server:app --app-dir src --port 8000 --reload`

...or, because I set up a custom Makefile target for it:

`make run`

**5. Open the front end**

Open `frontend/index.html` in your browser.  In VS Code you'll automatically be prompted to open this file when you execute step 3.  You can also go to VS Code's Ports window, right-click on the line for port 8000, and select "Open in Browser".

**6. Run the Solver**

Select a puzzle using the dropdown, and use the Play, Pause, and Step to execute the solver.

**7. Executing the unit tests**

Run this *from the /backend subdirectory*:

`pytest`

...or, using a custom make command I set up:

`make test`

A couple of the tests are marked as "integration tests" because they result in actual database queries instead of mocking them.  These tests are skipped during normal test execution.  To run the integration tests (only), run this *from the /backend subdirectory*:

`pytest -m integration`

...or again, using make:

`make itest`


CONCLUSION
----------
I'm *much* prouder of CrossCracker version 2 than version 1.  It's a substantially more impressive app in every way that matters, even if the only difference you can discern from watching the demos is that the puzzles it solves are larger.

I'm a little crestfallen that my friends and family are so unimpressed by it.  If you'd told me 10 years ago that I'd have the ability to write this app today, I wouldn't have believed you.  A computer program, written by one person in a few weeks, capable of solving a *New York Times crossword puzzle* in a matter of minutes?!?  This makes Deep Blue, the chess program that beat Garry Kasparov in 1996, look like kid stuff.  AI has made wonders like this commonplace.
