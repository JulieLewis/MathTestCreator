import os
from langchain.schema import Document
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from langchain.document_loaders import TextLoader,PyMuPDFLoader, CSVLoader, JSONLoader

class SourceMaterial():
    name: str
    source: str
    supported: bool
    types = ['text', 'image', 'video', 'audio', 'pdf', 'doc', 'ppt', 'xls', 'csv', 'json', 'xml', 'html']
    type: str
    metadata : dict
    length : int
    length_type : str
    documents: List[Document] = []
    GPT_MODEL = "gpt-4o-mini"

    def __init__(self, source):
        self.openai = OpenAI()
        self.load_source(source)
    
    def load_source(self, file):
        """
        Load the files in according to their type, prepare for chunking
        :param files: list of files to be loaded
        :return documents: list of documents containing page_content and metadata
        """
        try:
            self.supported = False
            self.source = file
            path = Path(file)
            self.type = path.suffix
            self.name = path.name
            if self.type in ['.txt', '.md']:
                loader = TextLoader(file)
                self.length_type = 'pages'
            elif self.type in ['.pdf']:
                self.supported = True
                loader = PyMuPDFLoader(file)
                self.length_type = 'pages'
            elif self.type in ['.csv']:
                loader = CSVLoader(file)
            elif self.type in ['.json']:
                loader = JSONLoader(file)
            else:
                # self.log(f"Loading file {file}")
                print(f"Unsupported file type {self.type}")
                self.unsupported_material("unsupported")
            document = loader.load()
            if len(document) > 0:
                self.metadata = document[0].metadata
                self.length = self.metadata['total_pages']
                self.documents.extend(document)
                self.get_document_summary(document)
        except Exception as e:
            print ('Upload Failed')

    def chat_response(self, model='gpt'):
        if model == 'gpt':
            response = self.openai.chat.completions.create(
            model= self.GPT_MODEL,
            messages=[{"role": "system", "content": "Provide a short summay of the document, no more than 30 words"},
                        {"role": "user", "content": f"Summarize the document: {sample}"}])
        self.summary = summary.choices[0].message.content or ""

    def unsupported_material(self):
        self.metadata = ""
        self.length = 0
        self.type = 'unsupported'
        self.summary = ""
        self.length_type = ""

    def get_document_summary(self, document):
        """
        Extract the data from the document
        """
        if len(document) > 15:
            sample = [page.page_content for page in document[:15]]
        else:
            sample =  [page.page_content for page in document]
        self.summary = 'test'
        # summary = self.openai.chat.completions.create(
        #     model= self.GPT_MODEL,
        #     messages=[{"role": "system", "content": "Provide a short summay of the document, no more than 30 words"},
        #                 {"role": "user", "content": f"Summarize the document: {sample}"}])
        # self.summary = summary.choices[0].message.content or ""

    def __repr__(self):
        return f"{self.name} - {self.summary}"

    