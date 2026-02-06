Remember that you only want the onnxruntime-gpu, not the base onnxruntime dependency.
The latter is sometimes a transparent dependency of other stuff like chromadb, so 
after an update, or refreshing the environment from requirements.txt from scratch, you
might have to uninstall both and reinstall only the -gpu version.


APP ARCHITECTURE
----------------
1. FastAPI
This app uses a FastApi framework for the "back end".  FastApi is basically a lightweight 
REST API framework similar to Flask.  But it's not actually a traditional client-server 
app that can run across the Internet.  Like, I couldn't put this on a server and let you 
run the interface in your browser.  Everything runs locally.  We're using this framework
mainly to cleanly separate the front end and make it easy to build a UI using well-known 
tools like HTML and JavaScript.  Making it Internet-capable is *possible* (I did it with
CrossCracker version 1)... but it's impractical.  It simply passes too much data back and
forth on every guess and would take forever to run as a web app.  It might be intersting 
to implement that feature at some point, but we'll cross that bridge if we ever come to it.


2. ChromaDB and RAG
The app uses a ChromaDB vector database to store a knowledge base of past crossword
clue-answer pairs.  It was populated using a dataset obtained from Hugging Face with
about 6.7 million rows.  Vector databases use a neural network to retrieve data "similar"
to the given input.  This makes them useful for all sorts of things, from helpbots to
medical diagnosis apps.  We use this database on startup to get a set of clue-answer
pairs that are "semantically similar" to each clue in the database.  In other words,
we get a list of "hints" for each clue.  We then pass those hints to an LLM to help it
select and rank a list of possible candidate answers for a clue.  This is classic RAG 
technology -- Retrieval Augmented Generation.

IMPORTANT: When installing a new environment with "pip install -r requirements.txt",
chromadb has dependencies on both onnxruntime and onnxruntime-gpu.  BUT THEY ARE
MUTUALLY EXCLUSIVE AND CONFLICT WITH EACH OTHER.  Not only that, but BOTH LIBRARIES
ARE OBSOLETE!  We only want the latest version of the -gpu version.  There are several 
workarounds to this, but the simplest is:
   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime-gpu

IMPORTANT: To use ChromaDB, you also need to install *both* the CUDA Toolkit *and*
cuDNN (the CUDA Deep Neural Network Library, which is built on top of the CUDA Tooklit).
These are not Python libraries, they are separate drivers.  They are available here:
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn
Be sure to get the right versions of everything!  At the time of this writing, we want
CUDA Toolkit 12.x and cuDNN 9.x.

IMPORTANT: ChromaDB is a memory hog.  To do anything, it has to load the entire set
of database indexes into memory.  That's just how vector DBs work.  In the case of our
6.4 million row crossword database, that index consumes about 11.5 GB of memory.
Yikes!!!  And it takes a long time to load the index the first time you run a query
(about 50 seconds on my system).  So I've set up a script to run chromaDB as a 
separate process.  Run this in an Ubuntu command prompt:
   chroma-up
To shut it down, run this in another Ubuntu command prompt:
   chroma-down
Or (a little more aggressive) press Ctrl+\ in the ChromaDB window, or (extremely 
aggressive) just close the window.

NOT IMPORTANT BUT INTERESTING: When using Chroma's preferred ONNXMiniLM_L6_V2 "embedding 
function" (which defines the mathematical procedure used to access the database), the 
call stack looks like this:
   app -> chromaDB library -> ONNX Runtime -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware
But if we use the SentenceTransformerEmbeddingFunction embedding function (which is said to
be slower AND more memory intensive, but has additional features -- which aren't 
actually used by ChromaDB, so we don't need them), the call stack would be:
   app -> chromaDB library -> PyTorch -> cuDNN -> CUDA Runtime -> CUDA Driver -> hardware
Hey, there's PyTorch!  Everyone loves PyTorch because it helped usher in the current age 
of AI programming.  But we're not using it.  It's slow and cumbersome.

ALSO INTERSTING: While ChromaDB's CUDAExecutionProvider has vastly superior performance
to the CPUExecutionProvider (as you would expect, since it uses the GPU instead of the
CPU), it doesn't really make any difference for our little crossword app.  For all the 
massive memory requirements ChromaDB imposes on us, in its entire run the solver executes 
just ONE database query which takes only a fraction of a second either way.
(Yes, one query...for efficiency you can batch queries together, so we get hints for 
every clue in the puzzle in just one call.)


3. Ollama and LLM
When we want to solve a particular crossword entry, we collect the clue, the "hints" we 
got from the vector database, and any other contextual information like the answer's
length and the current pattern of known letters.  We pass all this data to a local LLM
(managed by Ollama) along with a carefully constructed prompt instructing it to guess a
few answers and rank them with a confidence level from 0-100.  Currently we are using the
llama3.1:8b LLM, which basically has the intelligence of Elmer Fudd when compared to the 
LLMs you use online every day, but that's about all I can manage with the horsepower at
my disposal on my potato PC.  For comparison, there's a llama3.1:405b model available on
Ollama.  That's *50 times* as powerful (and 50 times the memory requirements) as the model
I'm using.  Sheesh!  Maybe one day I'll set up my app so it can call a powerful online 
LLM instead -- but that costs money, yo!

IMPORTANT: The LLM we're using needs 6-8 GB of RAM.  Combined with the 12 GB or so
required by the vector database, that puts a pretty heavy load on WSL's memory, which
I have capped at 20 GB.  Expect a lot of swapping and thrashing.


4. The Solver Algorithm
Believe it or not, all that stuff above with the AIs was the easy part.  I handled all 
that in just a few days, and most of THAT was faffing around with the LLM prompt.
(I cannot WAIT until they develop an API and make that crap obsolete.)

The hard part was actually the solver algorithm itself -- a good old-fashioned coding
problem.  Without getting into too much detail here, I'm using a variant of a 
"constraint satisfaction" algorithm with "dependency-aware backtracking".  What this 
means is that the app can guess an answer, and if it turns out to be wrong, it can 
backtrack and guess again, without getting stuck or taking years to solve a puzzle.

The main difference from the first version of the crossword solver is that the
solver guesses the answers for each clue at runtime.  I admit... I kind of cheated
in verson 1 of this app.  Yes, I used an LLM to get the answers.  HOWEVER, I retreved 
and recorded those answers before running the solver, and manually made sure they
included the correct answers.  That meant was that there was a small, fixed set of
posible answers for each clue when I ran the solver.

Removing that constraint and thus having an infinite set of possible answers per
clue is literally exponentially more difficult.  This meant that version 2 had to
be VASTLY better at guessing clues, even though it's still using the same LLM as 
version 1. Thanks to RAG technology, it is.

It also meant that the solving algorithm had to be much smarter about how to chose
which clues to answer, and more importantly, how and when to "backtrack" and try 
something different.  Against the advice of the AIs I consulted, I chose to model 
the algorithm after how humans (especially ME) solve crosswords IRL.  And it paid 
off!  I guess we humans still have a bit of an edge on the AIs after all...


5. Conslusion
I'm MUCH prouder of CrossCracker version 2 than version 1.  It's a much more
impressive app in every way that matters, even if you can't tell much of a difference
from watching the demos.  I'm a little crestfallen that my friends and family are so 
unimpressed by it.  AI has made wonders like this commonplace.  But if you'd told me
10 years ago that I'd have the ability to write this app today, I wouldn't have
believed you.  A computer program, written by one person in a few weeks, capable of 
solving a crossword puzzle in a matter of seconds?!?  Wake up, people, this is f-ing 
amazing!!!


Running the demo server (local development)
------------------------------------------

1. Install dependencies (FastAPI, uvicorn, etc.):

   python3 -m pip install -r requirements.txt

2. Start the server:

   uvicorn backend.src.server:app --port 8000

3. Open the front-end by opening `frontend/index.html` in your browser (or serve it with a static server).

Controls (UI): Play, Pause, Step, and Solve will control the solver and you'll see the grid update via WebSocket events.