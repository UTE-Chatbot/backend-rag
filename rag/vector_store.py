from langchain_core.prompts import ChatPromptTemplate
from data.loader import load_data
import re
import json
from models.gemini import Gemini
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
print(os.getenv("COHERE_API_KEY"))
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
        self.aug_docs = []  
        self.aug_docs_embs = []
        self.retrieve_top_k = 20
        self.embedding_model = embedding_model
        self.rerank_top_k = rerank_top_k
        self.index_name = index_name
        if self.is_indexed():
            self.idx = pc.Index(self.index_name)
        
    def generate_questions(text_chunk, num_questions=3):
        """
        Generates relevant questions that can be answered from the given text chunk.

        Args:
        text_chunk (str): The text chunk to generate questions from.
        num_questions (int): Number of questions to generate.
        model (str): The model to use for question generation.

        Returns: 
        List[str]: List of generated questions.
        """
        # Define the system prompt to guide the AI's behavior
        system_prompt = "Bạn là một trợ lý ảo chuyên nghiệp trong lĩnh vực tư vấn tuyển sinh. Nhiệm vụ của bạn là tạo ra các câu hỏi trọng tâm, chính xác, và chỉ dựa trên nội dung văn bản được cung cấp. Các câu hỏi cần tập trung vào những thông tin quan trọng và các khái niệm cốt lõi để người dùng có thể hiểu rõ hơn về nội dung."
        user_prompt = "Dựa vào đoạn văn bản sau, hãy tạo ra {numbers} câu hỏi ngắn gọn, rõ ràng, và chỉ có thể trả lời trực tiếp từ thông tin trong đoạn văn bản này: {chunk_content}. Trình bày câu trả lời dưới dạng danh sách các câu hỏi được đánh số, không cần thêm bất kỳ giải thích nào khác."

        # Generate questions using the OpenAI API
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                ("human", user_prompt),
            ]
        )


        model = Gemini(api_key=os.getenv("GEMINI_API_KEY"), llm_prompt_template=prompt)
        
        # Extract and clean questions from the response
        questions_text = model.chain.invoke({
            "numbers": num_questions,
            "chunk_content": text_chunk
        }).strip()
        questions = []
        
        # Extract questions using regex pattern matching
        for line in questions_text.split('\n'):
            # Remove numbering and clean up whitespace
            cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned_line and cleaned_line.endswith('?'):
                questions.append(cleaned_line)
        
        return questions

    def load_chunk(self, chunks) -> None:
        """
        Loads the data from the sources and partitions the HTML content into chunks.

        Parameters:
        chunks (list): A list of dictionaries representing the chunked documents, with 'title', 'text', and 'url' keys.
        """
        self.docs = chunks
        self.aug_docs = []
        # meta_data = json.loads(chunks[0]['meta_data'])
        # print(meta_data)

        # Loop index
        for i in range(len(chunks)):
            chunk = chunks[i]
            if (chunk["category"] == "FAQ"):
                continue
            print("Chunk:", chunk["content"])
            questions = self.generate_questions(chunk['content'])

            for question in questions:
                self.aug_docs.append({
                    "title": chunk['title'],
                    "content": chunk['content'],
                    "question": question,
                    EMBED_COLUMN_NAME: question,
                    "category": chunk['category'],
                    "sub_category": chunk['sub_category'],
                })


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
            print(">RUNNING BATCH", i)
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item[column] for item in batch]
            docs_embs_batch =  self.embedding_model.encode(texts)
            self.docs_embs.extend(docs_embs_batch)
        
        print("Document chunks embedded ✅")
            
        if len(self.aug_docs) == 0:
            return
        self.aug_docs_len = len(self.aug_docs)
        print("Aug docs len:", self.aug_docs[0])
        for i in range(0, self.aug_docs_len, batch_size):
            batch = self.aug_docs[i : min(i + batch_size, self.aug_docs_len)]
            texts = [item[column] for item in batch]
            aug_docs_embs_batch =  self.embedding_model.encode(texts)

            self.aug_docs_embs.extend(aug_docs_embs_batch)

        print("Questions chunks embedded ✅")

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
        # add type column to metadata
        for i in range(len(to_upsert)):
            to_upsert[i][2]['type'] = 'chunk'

        err = []
        for i in range(0, len(self.docs), batch_size):
            try: 
                i_end = min(i+batch_size, len(self.docs))
                self.idx.upsert(vectors=to_upsert[i:i_end])
            except Exception as e:
                print(f"Error upserting vectors: {e}")
                err.extend(to_upsert[i:i_end])

        print("Indexing complete for chunks")

        for aug_doc in self.aug_docs:
            aug_doc.pop(EMBED_COLUMN_NAME, None)
        meta_aug = self.aug_docs
        ids_aug = [str(i) for i in range(len(self.aug_docs))]
        to_upsert_aug = list(zip(ids_aug, self.aug_docs_embs, meta_aug))

        for i in range(len(to_upsert_aug)):
            to_upsert_aug[i][2]['type'] = 'question'

        print("TO UPSERT", to_upsert_aug)
        for i in range(0, len(self.aug_docs), batch_size):
            try: 
                i_end = min(i+batch_size, len(self.aug_docs))
                self.idx.upsert(vectors=to_upsert_aug[i:i_end])
            except Exception as e:
                print(f"Error upserting vectors: {e}")
                err.extend(to_upsert_aug[i:i_end])
        
        print("Indexing complete for questions")


        # let's view the index statistics
        print("Indexing complete")

        # Export json err 
        with open('error.json', 'w') as f:
            json.dump(err, f)

    def retrieve(self, query: str, is_logging: bool, top_k = 0) -> List[Dict[str, str]]:
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
        
        # Function to get unique matches
        def get_unique_matches(matches):
            seen_content = set()
            unique = []
            for match in matches:
                content = match['metadata'].get('content', '')
                if content not in seen_content:
                    seen_content.add(content)
                    unique.append(match)
            return unique

        # Try increasing top_k until we get enough unique results
        current_top_k = self.retrieve_top_k
        max_top_k = 100  # Maximum limit to prevent excessive queries
        unique_matches = []
        
        while current_top_k <= max_top_k and len(unique_matches) < self.retrieve_top_k:
            res = self.idx.query(vector=query_emb, top_k=current_top_k, include_metadata=True)
            unique_matches = get_unique_matches(res['matches'])
            current_top_k *= 2

        # If we still don't have enough matches, pad with the last unique match
        if len(unique_matches) < self.retrieve_top_k and unique_matches:
            last_match = unique_matches[-1]
            while len(unique_matches) < self.retrieve_top_k:
                unique_matches.append(last_match)
        
        # Ensure exactly retrieve_top_k matches
        res['matches'] = unique_matches[:self.retrieve_top_k]
        
        docs_to_rerank = [match['metadata']['content'] for match in res['matches']]
        print("Docs to rerank:", docs_to_rerank)
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
    # print("Vectorstore initialized ✅")
    # Delete all records
    # if vectorstore.is_indexed():
    #     print("Deleting all records...")
    #     pc.delete_index(vectorstore.index_name)
    #     print("All records deleted ✅")

    # Create new index
    # print("Creating new index...")
    # vectorstore.index()
    # print("New index created ✅")

    if not vectorstore.is_indexed():
        print("Indexing documents...")
        chunks = load_data(data_path, embed_columns)
        vectorstore.load_chunk(chunks[284:])
        print("Documents loaded and chunked ✅")
        vectorstore.embed()
        print("Documents embedded ✅")
        vectorstore.index()
        print("Vectorstore indexed ✅")

    return vectorstore
