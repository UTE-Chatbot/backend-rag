from data.loader import load_data
import json
import cohere
from embeddings import APIBaseEmbedding
from resources.constants import EMBED_COLUMN_NAME
from pinecone import Pinecone, PodSpec
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import os 
import dotenv

dotenv.load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# raw_documents = [
#         {
#             "title": "Effective prompt",
#             "url": "https://dkdt.hcmute.edu.vn/posts/610e39417da2ba0f77013765"}
# ]


class Vectorstore:
    """
    A class representing a collection of documents indexed into a vectorstore.

    Parameters:
    raw_documents (list): A list of dictionaries representing the sources of the raw documents. Each dictionary should have 'title' and 'url' keys.

    Attributes:
    raw_documents (list): A list of dictionaries representing the raw documents.
    docs (list): A list of dictionaries representing the chunked documents, with 'title', 'text', and 'url' keys.
    docs_embs (list): A list of the associated embeddings for the document chunks.
    docs_len (int): The number of document chunks in the collection.
    idx (hnswlib.Index): The index used for document retrieval.

    Methods:
    load_and_chunk(): Loads the data from the sources and partitions the HTML content into chunks.
    embed(): Embeds the document chunks using the Cohere API.
    index(): Indexes the document chunks for efficient retrieval.
    retrieve(): Retrieves document chunks based on the given query.
    """

    def __init__(self, index_name: str = "admission", rerank_top_k:int=3, embedding_model: APIBaseEmbedding = None) -> None:
        self.raw_documents = []
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.embedding_model = embedding_model
        self.rerank_top_k = rerank_top_k
        self.index_name = index_name
        if self.is_indexed():
            self.idx = pc.Index(self.index_name)

    def load_chunk(self, chunks) -> None:
        """
        Loads the data from the sources and partitions the HTML content into chunks.

        Parameters:
        chunks (list): A list of dictionaries representing the chunked documents, with 'title', 'text', and 'url' keys.
        """
        self.docs = chunks


    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            # Print html elements
            print(elements)
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self, column:str=EMBED_COLUMN_NAME) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item[column] for item in batch]
            docs_embs_batch =  self.embedding_model.encode(texts)
            self.docs_embs.extend(docs_embs_batch)
            
    def is_indexed(self) -> bool:
        """
        Checks if the documents are indexed.

        Returns:
        bool: True if the documents are indexed, False otherwise.
        """
        if self.index_name in pc.list_indexes().names():
            index = pc.Index(self.index_name)
            stats = index.describe_index_stats()
            total_records = stats.total_vector_count
            return total_records > 0
        else:
            return False
            

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        for doc in self.docs:
            doc.pop(EMBED_COLUMN_NAME, None)

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=len(self.docs_embs[0]),
                metric="cosine",
                spec=PodSpec(
                    environment="gcp-starter"
                )
                )

        # connect to index
        self.idx = pc.Index(self.index_name)

        batch_size = 128

        ids = [str(i) for i in range(len(self.docs))]
        # create list of metadata dictionaries
        meta = self.docs

        # create list of (id, vector, metadata) tuples to be upserted
        to_upsert = list(zip(ids, self.docs_embs, meta))

        err = []
        for i in range(0, len(self.docs), batch_size):
            try: 
                i_end = min(i+batch_size, len(self.docs))
                self.idx.upsert(vectors=to_upsert[i:i_end])
            except Exception as e:
                print(f"Error upserting vectors: {e}")
                err.extend(to_upsert[i:i_end])

        # let's view the index statistics
        print("Indexing complete")

        # Export json err 
        with open('error.json', 'w') as f:
            json.dump(err, f)

    def retrieve(self, query: str, is_logging: bool) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """
        print("Retrieving document for query: ", query)

        docs_retrieved = []
        query_emb = self.embedding_model.encode([query])[0]
        res = self.idx.query(vector=query_emb, top_k=self.retrieve_top_k, include_metadata=True)
        docs_to_rerank = [match['metadata']['content'] for match in res['matches']]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-v3.5",
        )

        docs_reranked = [res['matches'][result.index] for result in rerank_results.results]

        for doc in docs_reranked:
            docs_retrieved.append(doc['metadata'])

        if is_logging:
            summary = {}
            summary['query'] = query
            summary['docs_to_rerank'] = docs_to_rerank
            summary['rerank_results'] = []

            cnt = 0
            for doc in docs_reranked:
                cnt += 1
                summary['rerank_results'].append(f"{cnt}: {doc}")
            id = uuid.uuid4()
            with open(f'summary_{id}.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return docs_retrieved
    
def init_vectorstore(data_path: str, db_name: str, embed_columns:List[str], embedding_model:any, rerank_top_k) -> Vectorstore:
    vectorstore = Vectorstore(db_name, rerank_top_k, embedding_model)
    print("Vectorstore initialized ✅")

    if not vectorstore.is_indexed():
        print("Indexing documents...")
        chunks = load_data(data_path, embed_columns)
        vectorstore.load_chunk(chunks)
        print("Documents loaded and chunked ✅")
        vectorstore.embed()
        print("Documents embedded ✅")
        vectorstore.index()
        print("Vectorstore indexed ✅")

    return vectorstore
