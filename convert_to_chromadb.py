import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import logging
import glob
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KnowledgeBaseConverter:
    def __init__(self, persist_directory: str = "knowledge_base", knowledge_base_directory: str = "knowledge_base"):
        self.persist_directory = persist_directory
        self.knowledge_base_directory = knowledge_base_directory
        
        # Setup ChromaDB
        self.client = chromadb.PersistentClient(
            path=os.path.join(persist_directory, "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("code_vulnerability")
        
        # Model embedding
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.embedding_model.encode(text).tolist()

    def convert_to_chromadb(self):
        """Convert JSON files in knowledge base to ChromaDB"""
        knowledge_base_path = os.path.join(self.knowledge_base_directory)
        json_files = glob.glob(os.path.join(knowledge_base_path, "*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {knowledge_base_path}")
            return

        total_samples = 0
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    code_samples = json.load(f)
                if isinstance(code_samples, list) and code_samples:
                    logger.info(f"Converting {len(code_samples)} entries from {json_file}")
                    self._add_to_chromadb(code_samples)
                    total_samples += len(code_samples)
                else:
                    logger.warning(f"{json_file} is empty or invalid")
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
        
        logger.info(f"Converted a total of {total_samples} samples to ChromaDB")

    def _add_to_chromadb(self, code_samples: List[Dict]):
        """Add code samples to ChromaDB"""
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for sample in code_samples:
            if not all(key in sample for key in ["text", "label"]):
                logger.warning(f"Skipping invalid sample: {sample}")
                continue
            documents.append(sample["text"])
            embeddings.append(self._generate_embeddings(sample["text"]))
            metadatas.append({
                "label": str(sample.get("label", "0"))
            })
            ids.append(f"sample_{uuid.uuid4()}")
        
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} samples to ChromaDB collection")
        else:
            logger.warning("No valid samples to add to ChromaDB")

def main():
    print("""
    ====================================
     KNOWLEDGE BASE TO CHROMADB CONVERTER
    ====================================
    """)
    
    try:
        converter = KnowledgeBaseConverter()
        converter.convert_to_chromadb()
        print("\nConversion completed successfully!")
        count = converter.collection.count()
        print(f"ChromaDB now contains {count} documents")
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()