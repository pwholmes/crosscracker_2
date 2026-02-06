Installing and running the app
------------------------------
1. Install dependencies (FastAPI, uvicorn, etc.).
See notes below about onnxruntime:

   python3 -m pip install -r requirements.txt
   python3 -m pip uninstall onnxruntime onnxruntime-gpu
   python3 -m pip install onnxruntime-gpu

2. Populate the RAG database.  
I haven't uploaded the database file to GitHub because it's too big, but you can create and populate the database from scratch from the HuggingFace dataset I used by running /scripts/load_dataset_1.py.  The database will be created in the /db directory.  The script will download the dataset from the HF website and jam it into the database.  It's free, you just need an access token that you get from the website.  Put the token in a .env file in your project root:

    HF_TOKEN=mykey
    
You might have to mess around with the path to the .env file because the script is in a subdirectory, I'm not sure.  Anyway, a full load took about 6 hours on my system, so... have fun!

3. Start ChromaDB.
The app expects Chroma to be running in a separate process on local port 8001.  Here's the "chroma-up" script I created to start it:

   CHROMA_EXE_DIR=/home/pwholmes/dev/crosscracker_2/.venv/bin
   CHROMA_DB_DIR=/home/pwholmes/dev/crosscracker_2/backend/db
   echo "Starting ChromaDB server from $CHROMA_DB_DIR..."
   $CHROMA_EXE_DIR/chroma run --path "$CHROMA_DB_DIR" --port 8001

For what it's worth, here's "chroma-down".  Not sure kill is "graceful", but it's probably better than forcefully closing the window Chroma is running in, or using Ctrl+/:

   #!/bin/bash
   PID=$(lsof -t -i:8001)
   if [ -z "$PID" ]; then
      echo "No Chroma server found on port 8001."
   else
      echo "Gracefully stopping Chroma (PID $PID)..."
      kill $PID
   fi

Note that this database requires about 12 GB of RAM.  Chroma allocates the memory the first time you run a query against the database (i.e., not when you start Chroma), and keeps it allocated until you stop Chroma.

4. Start the LLM
I'm using Ollama with the llama3.1:8b LLM, which runs in 6-8 GB of RAM.  Feel free to use a bigger/better LLM if you want.  There's a constant in llm.py where you set the model name.  Run the LLM using:

   ollama run llama3.1:8b

5. Start the server:

   uvicorn backend.src.server:app --port 8000

6. Open the front-end by opening `frontend/index.html` in your browser (or serve it with a static server).

Select a puzzle using the dropdown, and use the Play, Pause, Step, and Solve buttons to solve the puzzle.  You'll see the grid update via WebSocket events as the app solves the puzzle.  Solve was supposed to be a "fast" solution that doesn't pause between steps, but the app is
now bound by how long it takes to run the LLM queries, which is much longer than the .25 second delay I put in the UI to pause beteen steps so you can watch things happening when you press Play.  Solve is only useful for test puzzles where we mock calls to the LLM, to exercise the solver logic.


APP ARCHITECTURE
----------------
1. FastAPI
This app uses a FastApi framework for the "back end".  FastApi is basically a lightweight REST API framework similar to Flask.  But CrossCracker not actually a traditional client-server app that can run across the Internet.  I couldn't put the back end on an AWS server and then run the interface in a browser on my phone.  Everything runs locally.  We're using FastAPI mainly to cleanly separate the front end and make it easy to build a UI using well-known tools like HTML and JavaScript.  Making it Internet-capable is *possible* (I did it with CrossCracker version 1)... but it's impractical.  It simply passes too much data back and forth on every guess and would take forever to run as a web app.  It might be intersting to implement that feature at some point, but we'll cross that bridge if we ever come to it.


2. ChromaDB and RAG
The app uses a ChromaDB vector database to store a knowledge base of past crossword clue-answer pairs.  It was populated using a dataset obtained from Hugging Face with about 6.4 million rows.  Vector databases use a neural network to retrieve data "similar" to the given input.  This makes them useful for all sorts of things, from helpbots to medical diagnosis apps.  In this app, we use the vector database on startup to get a set of clue-answer pairs that are "semantically similar" to each clue in the database.  In other words, we get a list of "hints" for each clue.  We then pass those hints to an LLM to help it select and rank a list of possible candidate answers for a clue.  This is classic RAG technology -- Retrieval Augmented Generation.


IMPORTANT: When installing a new environment with "pip install -r requirements.txt", chromadb has dependencies on both onnxruntime and onnxruntime-gpu.  BUT THEY ARE MUTUALLY EXCLUSIVE AND CONFLICT WITH EACH OTHER.  Not only that, but BOTH LIBRARIES IT INSTALLS ARE OBSOLETE!  We only want the latest version of the -gpu version.  There are several workarounds to this, but the simplest is to run this after installing the requirements.txt:

   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime-gpu


IMPORTANT: To use ChromaDB, you also need to install *both* the CUDA Toolkit *and* cuDNN (the CUDA Deep Neural Network Library, which is built on top of the CUDA Tooklit).  These are not Python libraries, they are separate executables.  They are available here:

   https://developer.nvidia.com/cuda-downloads
   https://developer.nvidia.com/cudnn

Be sure to get the right versions of everything!  At the time of this writing, we want CUDA Toolkit 12.x and cuDNN 9.x.


IMPORTANT: ChromaDB is a memory hog.  To do anything, it has to load the entire database index into memory.  That's just how vector DBs work.  In the case of our 6.4 million row crossword database, that index consumes about 11.5 GB of memory.  Yikes!!!  And it takes a long time to load the index the first time you run a query (about 50 seconds on my system).  So I've set up a script to run chromaDB as a separate process.  Run this in an Ubuntu command prompt:

   chroma-up

To shut it down, run this in another Ubuntu command prompt:

   chroma-down

Or (a little more aggressive) press Ctrl+\ in the ChromaDB window, or (extremely aggressive) just close the window Chroma is running in.


NOT IMPORTANT BUT INTERESTING: With the code using Chroma's preferred ONNXMiniLM_L6_V2 "embedding function" (which defines the mathematical procedure used to access the database), the call stack looks like this:

   app -> chromaDB library -> ONNX Runtime -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware

But if it uses the SentenceTransformerEmbeddingFunction embedding function (which is said to be slower AND more memory intensive, but has additional features -- which aren't actually used by ChromaDB, so we don't need them), the call stack would be:

   app -> chromaDB library -> PyTorch -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware

Hey, there's PyTorch!  Everyone loves PyTorch because it helped usher in the current age of AI programming.  But we're not using it.  It's slow and cumbersome and unnecessary.


ALSO INTERSTING: While ChromaDB's CUDAExecutionProvider has vastly superior performance to the CPUExecutionProvider (as you would expect, since it uses the GPU instead of the CPU), it doesn't really make any difference for our little crossword app.  For all the massive memory requirements ChromaDB imposes on us, in its entire run the solver executes just ONE database query which takes only a second or two, regardless of whether you're using the GPU.


3. Ollama and LLM
When we want to solve a particular crossword entry, we collect the clue, the "hints" we got from the vector database, and any other contextual information like the answer's length and the current pattern of known letters.  We pass all this data to a local LLM (managed by Ollama) along with a carefully constructed prompt instructing it to guess a few answers.  Then we call the LLM again and ask it to score these answers with a confidence level from 0-100.

Why two separate calls, you ask?  Because LLMs are dumb.  If you ask them to rank their own answers in the first call, they'll rank everything at or near 100%.  Remember, the LLM *chose* those answers because they satisfied its own internal criteria, so in the LLM's opinion they're all good answers.  Asking it to rank them in a separate call forces it to consider them more carefully and you get much better results.

Currently we're using the llama3.1:8b LLM, which basically has the intelligence of Elmer Fudd when compared to the LLMs you use online every day, but that's about all I can manage with the horsepower at my disposal on my potato PC.  For comparison, there's a llama3.1:405b model available on Ollama.  That's *50 times* as powerful (and 50 times the memory requirements) as the model I'm using.  Sheesh!  Maybe one day I'll set up my app so it can call a powerful online LLM instead -- but that costs money, yo!

IMPORTANT: The LLM we're using needs 6-8 GB of RAM.  Combined with the 12 GB or so required by the vector database, that puts a pretty heavy load on WSL's memory, which I have capped at 20 GB.  Expect a lot of swapping and thrashing.


4. The Solver Algorithm
Believe it or not, all that stuff above with the AIs was the easy part.  I handled all that in just a few days, and most of THAT was faffing around with the LLM prompt.  (I cannot WAIT until they develop an API and make that crap obsolete.)

The really hard part was the solver algorithm itself -- a good old-fashioned coding problem.  Without getting into too much detail here, I'm using a variant of a "constraint satisfaction" algorithm with "dependency-aware backtracking".  What this means is that the app can guess an answer, and if it turns out to be wrong, it can backtrack and guess again, without getting stuck, going into an infinite loop, or taking longer than the lifetime of the universe to solve a puzzle.

The main difference from the first version of the crossword solver is that the solver guesses the answers for each clue at runtime.  I admit... I kind of cheated in verson 1 of this app.  Yes, I used an LLM to get the answers, HOWEVER, I ran the LLM and recorded its answers before running the solver app, and made sure the *correct* answers were in the list.  That meant there was a small, fixed set of possible answers for each clue when I ran the solver.  What you see in the demo is just the solver algorithm -- AI isn't even involved anymore by that point.

Letting the LLM generate guesses in real time while the app runs, and thus having an effectively infinite set of possible answers per clue, is literally an exponentially more difficult problem.  Version 2 had to be VASTLY better at guessing clues, even though it's still using the same LLM as version 1.  Thanks to RAG technology, it is.  It usually gets about 90% of the clues in the test puzzles correct on the first guess.

Still, the solving algorithm also had to be much smarter about how to chose the order in which to answer clues, and more importantly, how and when to "backtrack" and try something different when it gets stuck.  I chose to model the algorithm after how humans (like me!) solve crosswords IRL.  Frankly, the AIs I consulted -- and I asked all of the big ones -- were absolutely hopeless at this task.  Even with a lot of hand-holding, the best AIs out there are still completely incapable of properly solving a problem of this complexity.  I guess we humans still have a bit of an edge on the AIs after all.  For now, anyway...


5. Conslusion
I'm MUCH prouder of CrossCracker version 2 than version 1.  It's a much more impressive app in every way that matters, even if you can't tell much of a difference from watching the demos.  I'm a little crestfallen that my friends and family are so unimpressed by it.  AI has made wonders like this commonplace.  But if you'd told me 10 years ago that I'd have the ability to write this app today, I wouldn't have believed you.  A computer program, written by one person in a few weeks, capable of solving a crossword puzzle in a matter of seconds?!?  Wake up, people, this is f-ing amazing!!!
