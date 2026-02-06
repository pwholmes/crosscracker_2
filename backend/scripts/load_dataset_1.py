from huggingface_hub import login
from datasets import Dataset, load_dataset # type:ignore
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_hf_dataset():
    # Load the first dataset.  If we don't have it already, it is downloaded from the HuggingFace 
    # website.  After the first execution it is cached in ~/.cache/huggingface/datasets.  
    # (It checks for updates even if we do have it, which is why we call login() even on subsequent
    # executions.)
    # The login() parameter is a token obtained from the HuggingFace website.
    # It isn't accessible from the website after creation, so if you somehow lose this one you'll 
    # need to create a new one.
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set. Export your Hugging Face token before running.")
    login(hf_token)
    # The dataset has two "splits": "train" and "validate", whose purpose is just what you'd imagine.
    # This dataset has three columns: id, clue, and answer.
    print("Opening dataset...")
    ds = load_dataset("albertxu/CrosswordQA", split="train")

    # Filter out rows where 'answer' is missing or just whitespace
    print("Filtering dataset...")
    ds = ds.filter(lambda x:  #type: ignore
        x["clue"] is not None and str(x["clue"]).strip() != "" and #type: ignore
        x["answer"] is not None and str(x["answer"]).strip() != "") #type: ignore

    # Now insert the data into the ChromaDB database.
    # Explicitly request the CUDA provider
    print("Opening database...")
    client = chromadb.PersistentClient(path = "../db")
    gpu_ef = ONNXMiniLM_L6_V2(preferred_providers=['CUDAExecutionProvider'])
    collection = client.get_or_create_collection(name="crossword_1", embedding_function=gpu_ef) #type: ignore
    
    BATCH_SIZE:int = 5000
    total_rows:int = len(ds)
    checkpoint_file:str = "../db/last_index_1.txt"
    start_index:int = 0

    # Read where we last finished
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            content = f.read().strip()
            if content:
                start_index = int(content)
                print(f"Resuming from row {start_index}")
    
    # Insert the dataset into the database.
    for i in range(start_index, total_rows, BATCH_SIZE):
        print(f"Inserting rows {i}-{i+BATCH_SIZE-1}...")
        start = time.perf_counter()
        batch:Dataset = ds.select(range(i, min(i+BATCH_SIZE, total_rows))) #type: ignore
        
        print("Converting data...")
        ids: list[str] = [str(x) for x in batch["id"]]  # type: ignore
        clues: list[str] = list(batch["clue"]) #type: ignore
        answers:list[dict] = [{"answer": ans} for ans in batch["answer"]]  # type: ignore

        try:        
            print("Starting batch insertion...")
            collection.upsert(ids=ids, documents=clues, metadatas=answers) #type: ignore
            end = time.perf_counter()

            # Note that the value we are writing is the start index of the NEXT batch
            print("Updating checkpoint...")
            with (open(checkpoint_file, "w")) as f:
                f.write(str(i+len(ids)))
            
            print(f"Batch complete.  Elapsed time: {end-start:.6f} seconds")
        except Exception as e:
            print(f"Error occurred at index {i}: {e}")
            break

if __name__ == "__main__":
    # I do NOT want to run this again accidentally!
    # It takes several (6+) hours to reload the entire dataset into the chromaDB vector database.
    # Make ABSOLUTELY sure you know what you're doing before you un-comment this.
    print ("Loading HF dataset...")
    #load_hf_dataset()
