# pyright: basic
import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
import time
from pathlib import Path
from .llm import LLM
from .model import Entry, Cell


def open_database() -> Collection:
    print("Opening database...")

    # The PersistentClient opens the database in the app's own memory space.
    # The HttpClient assumes the server is running in another process, which is generally
    # preferable because it consumes a crapload of memory and takes a long time to start up
    # (about 50 seconds).  I set up a script to start the database from an Ubuntu
    # command prompt.  Just run chroma-up.
    # Remember that either way, the first time you run a query it will still be slow while
    # it loads up the database index.  But when you run it as a separate server process, 
    # you don't have to wait for that every time you run this app.
    # BTW, you can close the database server with Ctrl+\ or just closing the terminal window.
    #db_path = Path(__file__).resolve().parent.parent / "db"
    #client = chromadb.PersistentClient(path=str(db_path))
    client = chromadb.HttpClient(host='localhost', port=8001)

    # The SentenceTransformer embedding function is allegedly much faster than the 
    # ONNXMiniLM_L6_V2 embedding function, at the cost of using much more memory.  Unfortunately 
    # an embedding function is basically a mathematical algorithm for storing data in a vector
    # database, so once you've created a DB with one, you can't switch to another.  Seeing as how
    # it took all day to create this database using ONNXMiniLM_L6_V2, and performance isn't likely
    # to be an issue with this little toy app, I'm not eager to start over.  Just keep it in mind
    # for next time!  (Note that the device="cuda" parameter is essential if you want it to use
    # the GPU instead of the CPU.)
    #print(ort.get_available_providers())
    #gpu_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")
    gpu_ef = ONNXMiniLM_L6_V2(preferred_providers=['CUDAExecutionProvider'])

    collection: Collection = client.get_or_create_collection(name="crossword_1", embedding_function=gpu_ef) #type: ignore
    return collection

def query_database(collection: Collection, entries: list[Entry]):
    # The parameters to collection.query() are:
    #   query_texts: the list of crossword clues.
    #   n_results: the maximum number of result sets per query.
    # The return values are:
    #   results["documents"]: the "example clues" found in the database.
    #     In chromaDB, each record is a "document" (just a string, really) with associated 
    #     "metadatas".  In our case, each document is a crossword clue.
    #     This value is a list of lists of strings: n_results clues for each clue 
    #     in the query.
    #     For example, if the clue in the query was "Old-fashioned butter maker" and 
    #     n_results was set to 8, the list of example clues retrieved from the database 
    #     might include "Butter-maker", "Butter maker's need", and 6 other similar clues.
    #   results["metadatas"]: the "example answers" for the example clues.
    #     This value is a list of lists of dictionaries: n_results answers for each clue 
    #     in the query.  But instead of a simple string, it's a dictionary with one entry,
    #     keyed by "answer".  (That's because "answer" is the only metadata we provided for 
    #     each document record when we populated the database.)
    #     For example, if the the clue in the query was "Old-fashioned butter maker" and 
    #     n_results was set to 8, the list of example metadatas retrieved from the database
    #     might include {"answer": "churn"}, {"answer": "churners"}, and 6 other similar
    #     answers.
    print("Running query...")
    clues = [entry.clue for entry in entries]
    results = collection.query(query_texts=clues, n_results=8)

    # Process the results
    docs = results.get("documents")
    metas = results.get("metadatas")
    if results is None or docs is None or metas is None:
        raise ValueError("Query failed: No results returned, or results are missing data.")
    if len(docs) != len(entries) or len(metas) != len(entries):
        raise ValueError("Query failed: Number of results does not equal number of queries.")
    
    # Pack the result data into more compact, readable tuples and store them in the crossword_entries.
    for i in range(len(entries)):
        hints_list: list[tuple[str, str]] = []
        for example_clues, example_answers in zip(docs[i], metas[i]):
            normalized_answer = "".join(c for c in str(example_answers["answer"]) if c.isalpha()).upper()
            hints_list.append((str(example_clues), normalized_answer))
        entries[i].hints = hints_list

def populate_hints(entries: list[Entry]) -> None:
    """Populate `Entry.hints` by querying the vector database."""
    collection: Collection = open_database()
    query_database(collection, entries)


def get_answer(entries: list[Entry]):
    start = time.perf_counter()

    # Open the clue database
    collection: Collection = open_database()

    # Query the clue database.
    query_database(collection, entries)

    # Fashion an appropriate LLM prompt for each clue using the available context data.
    for entry in entries:
        prompt = LLM.create_prompt(entry, 0)
        print(prompt)

    end = time.perf_counter()
    print(f"Elapsed time: {end-start}")
