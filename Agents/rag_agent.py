from agents.agent import Agent
from openai import OpenAI
from langchain.schema import Document
from typing import List, Dict
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader,PyPDFLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGAgent(Agent):

    GPT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    DB = "vector_db"
    COLLECTION_NAME = "math_documents"
    
    def __init__(self):
        """
        Set up this instance by connecting to OpenAI, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing RAG Agent")
        self.openai = OpenAI() 
        self.document_dict = {}
        self.log("Agent is ready")

    def create_init_collection(self) -> chromadb.Collection:
        client = chromadb.PersistentClient(path=self.DB)
        existing_collection_names = [collection.name for collection in client.list_collections()]
        if self.COLLECTION_NAME in existing_collection_names:
            client.delete_collection(self.COLLECTION_NAME)
            self.log(f"Deleted existing collection: {self.COLLECTION_NAME}")

        self.collection = client.create_collection(self.COLLECTION_NAME)

    def load_files(self, files: List[str]) -> List[Document]:
        """
        Load the files in according to their type, prepare for chunking
        :param files: list of filea to be loaded
        :return documents: list of documents containing page_content and metadata
        """
        self.documents=[]
        for file in files:

            file_type = Path(file).suffix
            if file_type in ['.txt', '.md']:
                loader = TextLoader(file)
            elif file_type in ['.pdf']:
                loader = PyPDFLoader(file)
            elif file_type in ['.csv']:
                loader = CSVLoader(file)
            elif file_type in ['.json']:
                loader = JSONLoader(file)
            else:
                self.log(f"Loading file {file}")
                raise ValueError(f"Unsupported file type {file_type}")
            document = loader.load()
            if len(document) > 0:
                self.get_document_data(document)
                # print(f"loading document {document[0]}")
                self.documents.extend(document)


    
    def recursive_text_splitter(self) -> List[Document]:
        """
        Split the loaded documents into chunks of text ready to be embedded for a vector database
        :param documents: list of documents containing page_content and metadata
        :return chunks: list of chunks of textcontaining page_content and metadata
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = text_splitter.split_documents(self.documents)
    
    def get_document_data(self, document) :
        """
        Extract the data from the document
        """
        print(f"loading document")
        self.document_dict[document[0].metadata['source']] = {}
        self.document_dict[document[0].metadata['source']]['pages']= max([page.metadata['page'] for page in document])
        sample = [page.page_content for page in document[:15]]
        summary = self.openai.chat.completions.create(
            model= self.GPT_MODEL,
            messages=[{"role": "system", "content": "Provide a short summay of the document, no more than 50 words"},
                        {"role": "user", "content": f"Summarize the document: {sample}"}])
        self.document_dict[document[0].metadata['source']]['summary'] = summary.choices[0].message.content
        print(self.document_dict)
        return self.document_dict
        # return f"The documents contains {pages} pages. This is a pdf about {result}"
    def create_collection(self) -> chromadb.Collection:
        """
        Create a collection in the Chroma database and add the chunks to it
        :param chunks: list of chunks of textcontaining page_content and metadata
        :return collection: Chroma collection containing the chunks
        """

        for i, chunk in enumerate(self.chunks):
            self.collection.add(
                ids=[f"doc_{i}"],
                documents=[chunk.page_content],
                embeddings=[self.EMBEDDING_MODEL.encode(chunk.page_content)],
                metadatas=[chunk.metadata]
            )
        self.log(f"Added {len(self.chunks)} chunks to collection: {self.collection.name}")
    def setup(self) -> chromadb.Collection:
        self.create_init_collection()
        self.recursive_text_splitter()
        self.create_collection()
        return self.collection
    
    def messages_for(self, question: str) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param question: a question you want to ask
        :param related: related passages in the documents provided

        :return: the list of messages in the format expected by OpenAI
        """
        system_message = "You write a question for a math test. If you are not sure say you don't know."
        user_prompt = "The subject of the question is: "
        user_prompt += question
        user_prompt += "To make context for the question, here is a piece of text from a math textbook:"
        for related in self.find_related(question):
            user_prompt += related
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "The question you are writing is:"},
    ]

    def find_related(self, description: str):
        """
        Return a list of text related to the given one by looking in the Chroma datastore
        """
        self.log("Agent is performing a RAG search of the Chroma datastore to find related text")
        vector = self.EMBEDDING_MODEL.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        related = results['documents'][0][:]
        self.log("Agent has found related text")
        return related

    def gpt_4o_mini_rag(self, question: str):
        """
        Use the RAG model to stream result to a question
        """
        related = self.find_related(question)
        stream = self.openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=self.messages_for(question),
            stream=True
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result
        return result
        