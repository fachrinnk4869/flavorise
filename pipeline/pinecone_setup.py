# pipeline/pinecone_setup.py
import json
import os
import asyncio
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from pinecone import ServerlessSpec
from get_embedding import get_dense_embeddings
from pathlib import Path

# load env
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
NAME_PINECONE_DENSE = os.getenv('NAME_PINECONE_DENSE')
NAME_PINECONE_SPARSE = os.getenv('NAME_PINECONE_SPARSE')
NAMESPACE = os.getenv('NAMESPACE')
NAMESPACE2 = os.getenv('NAMESPACE2')
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024

# config
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index():
    if not pc.has_index(NAME_PINECONE_DENSE):
        print("create dense index")
        pc.create_index(
            name=NAME_PINECONE_DENSE,
            vector_type="dense",
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    if not pc.has_index(NAME_PINECONE_SPARSE):
        print("create sparse index")
        pc.create_index(
            name=NAME_PINECONE_SPARSE,
            vector_type="sparse",
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            )
        )

def generate_embedding(path_files, bm25_model, column='text'):
    try:
        with open(path_files, 'r') as file:
            data = json.load(file)
            dense_vectors = []
            sparse_vectors = []
            for item in data:
                # get dense embedding
                dense_item = {
                    "id": item['id'], 
                    "values": get_dense_embeddings(item[column], EMBED_DIM), 
                    # "metadata": {key: value for key, value in item.items() if key in {"category"}}
                    # "metadata": item['all_text']
                }
                if dense_item["values"] is not None:
                    dense_vectors.append(dense_item)
                # get sparse embedding
                sparse_vals = bm25_model.encode_documents([item[column]])[0]
                if sparse_vals and sparse_vals.get("indices") and sparse_vals.get("values"):
                    sparse_item = {
                        "id": item["id"],
                        "sparse_values": sparse_vals,
                        # "metadata": {k: v for k, v in item.items() if k in {"category"}}
                        # "metadata": item['all_text']
                    }
                    sparse_vectors.append(sparse_item)
            
            return dense_vectors, sparse_vectors
    except FileNotFoundError:
        print(f"Error: {path_files} not found. Please ensure the file exists in the correct directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path_files}. The file might be malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# read data
folder_path = 'data/clean'

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
BM25_DIR = BASE_DIR / "pipeline" / "model"
BM25_DIR.mkdir(parents=True, exist_ok=True)
BM25_PATH = BM25_DIR / "bm25_params.json"
BM25_PATH2 = BM25_DIR / "bm25_params_all.json"

def create_corpus(corpus, folder_path, column='text'):
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        for item in data:
                            corpus.append(item[column])
                        
                except FileNotFoundError:
                    print(f"Error: {file_path} not found. Please ensure the file exists in the correct directory.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {file_path}. The file might be malformed.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
    
    print("corpus created successfully")

# define bm25 model
def create_corpus_train_bm25_model(bm25, folder_path, column='text'):
    # folder_path juga dibuat absolut
    folder_path = (BASE_DIR / folder_path).resolve()
    bm25_corpus = []
    create_corpus(bm25_corpus, str(folder_path), column)
    bm25.fit(bm25_corpus)
    # ingredient text only
    if(column == 'text'):
        bm25.dump(str(BM25_PATH))
    else: # all text
        bm25.dump(str(BM25_PATH2))

    print("bm25 model successfully loaded")

# helper to chunk vector
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def main():
    # create index (if not available)
    create_index()
    # get index vector db
    index_dense = pc.Index(name=NAME_PINECONE_DENSE)
    index_sparse = pc.Index(name=NAME_PINECONE_SPARSE)
    # create corpus and train bm25 model
    bm25 = BM25Encoder(stem=False)
    # # create corpus for ingredient text only
    # create_corpus_train_bm25_model(bm25, folder_path)
    
    # CREATE CORPUS FOR ALL TEXT
    create_corpus_train_bm25_model(bm25, folder_path, 'all_text') 
    print("load bm25 model done")

    # generate dense and sparse vector
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                """
                GENERATE EMBEDDING AND UPSERT FROM TEXT INGREDIENT ONLY
                """
                # # insert data dense
                # dense_vectors, sparse_vectors = generate_embedding(file_path, bm25_model=bm25)
                # print("generate dense and sparse vector done for: ", filename)
                # # upsert dense per 100
                # for idx, batch in enumerate(chunked(dense_vectors, 100)):
                #     index_dense.upsert(
                #         vectors=batch,
                #         namespace=NAMESPACE
                #     )
                #     print(f"successfully insert {(idx+1)*100} data")
                # print("upsert dense vector successfully for:", filename)

                # # upsert sparse per 100
                # for idx, batch in enumerate(chunked(sparse_vectors, 100)):
                #     index_sparse.upsert(
                #         vectors=batch,
                #         namespace=NAMESPACE
                #     )
                #     print(f"successfully insert {(idx+1)*100} data")
                # print("upsert sparse vector successfully for:", filename)
                """
                GENERATE EMBEDDING AND UPSERT FROM TITLE+INGREDIENT+STEP TEXT
                """
                # insert data dense
                dense_vectors, sparse_vectors = generate_embedding(file_path, bm25_model=bm25, column='all_text')
                print("(all_text) generate dense and sparse vector done for: ", filename)
                # upsert dense per 100
                for idx, batch in enumerate(chunked(dense_vectors, 100)):
                    index_dense.upsert(
                        vectors=batch,
                        namespace=NAMESPACE2
                    )
                    print(f"successfully insert {(idx+1)*100} data")
                print("(all_text) upsert dense vector successfully for:", filename)

                # upsert sparse per 100
                for idx, batch in enumerate(chunked(sparse_vectors, 100)):
                    index_sparse.upsert(
                        vectors=batch,
                        namespace=NAMESPACE2
                    )
                    print(f"successfully insert {(idx+1)*100} data")
                print("(all_text) upsert sparse vector successfully for:", filename)

if __name__ == "__main__":
    main()