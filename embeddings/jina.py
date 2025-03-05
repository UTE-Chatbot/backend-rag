

import requests
import json

import os
from typing import Optional, Union, List
from embeddings import APIBaseEmbedding

class JinaEmbedding(APIBaseEmbedding):
    def __init__(
            self,
            name: str = "jina-clip-v2",
            apiKey: str = None,
        ):
        super().__init__(name=name, apiKey=apiKey)

        self.url = "https://api.jina.ai/v1/embeddings"
        self.apiKey = apiKey or os.getenv("JINA_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.apiKey}"
        }

    def encode(self, docs: List[str]):
        try:
            data = {
                "model": self.name,
                "dimensions": 1024,
                "normalized": True,
                "embedding_type": "float",
                "input": docs
            }

            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            json_data = response.json()
            data = json_data["data"]
            embeddings = []
            for item in data:
                embeddings.append(item["embedding"])

            return embeddings

        except Exception as e:
            raise ValueError(
                f"Failed to get embeddings. Error details: {e}"
            ) from e