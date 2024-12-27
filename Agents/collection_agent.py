from agents.agent import Agent
from openai import OpenAI
from langchain.schema import Document
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from agents.source_material import SourceMaterial
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents.test_question import Question, TestQuestions

class CollectionAgent(Agent):

    GPT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    DB = "vector_db"
    COLLECTION_NAME = "math_documents"
    
    DraftTest: TestQuestions
    FinalTest: TestQuestions
    restricted_questions: List

    def __init__(self):
        """
        Set up this instance by connecting to OpenAI, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing RAG Agent")
        self.openai = OpenAI() 
        self.restricted_questions = []
        self.materials = []
        self.chunks = []
        self.collection = ''
        self.FinalTest = ''
        self.log("Agent is ready")

    def create_source_material(self, sources):
        """
        Create a SourceMaterial object from a list of sources
        :param sources: list of sources to be loaded
        """
        try:
            for source in sources: 
                if source not in [doc.source for doc in self.materials]:
                        self.materials.append(SourceMaterial(source))
                        print('file added')
                        return SourceMaterial(source)
                else:
                    print ('already uploaded')
        
        except Exception as e:
            "Print NO Source"

    def materials_rag(self, material_selections):
        markdown_output = ''
        for i, material in enumerate(self.materials):
            min_page_init = max(int(material_selections.loc[i]['Start']),0) #greater than 0
            max_page_init = min(int(material_selections.loc[i]['End']), material.length) #shorten than total pages (not necessary)
            min_page = min(min_page_init,material.length) # not bigger than max length
            max_page = max(max_page_init,min_page) # bigger than start
            selected_documents = [doc for doc in material.documents if min_page <= doc.metadata['page'] <= max_page]
            self.recursive_text_splitter(selected_documents)

            markdown_output += f"### Material {material_selections.loc[i]['Name']} has been added from {material_selections.loc[i]['Unit']} {min_page} to {max_page}\n\n"
        yield markdown_output


    def recursive_text_splitter(self, material):
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
        self.chunks.extend(text_splitter.split_documents(material))

    def create_init_collection(self) -> chromadb.Collection:
        """
        Intialize the chroma DB collection
        Delete existing collection if it exists
        """

        client = chromadb.PersistentClient(path=self.DB)
        existing_collection_names = [collection.name for collection in client.list_collections()]
        if self.COLLECTION_NAME in existing_collection_names:
            client.delete_collection(self.COLLECTION_NAME)
            self.log(f"Deleted existing collection: {self.COLLECTION_NAME}")

        self.collection = client.create_collection(self.COLLECTION_NAME)
        print('initial collection set')

    def add_to_collection(self) -> chromadb.Collection:
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
        self.add_to_collection()
        # return self.collection

    # def chat_response(self,messages, model):
    #     """
    #     Make a call to a frontier model and return text
    #     :param system message: system message for model
    #     :param user_prompt: user prompt for message
    #     :param model: model to be used as switch 
    #     """
    #     if model == 'gpt':
    #         response = self.openai.chat.completions.create(
    #         model= self.GPT_MODEL,
    #         messages=messages)
    #         return response.choices[0].message.content or ""
            
    # def stream_response(self,system_message, user_prompt, model='gpt'):
    #     messages = [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_prompt},
    #         {"role": "assistant", "content": "The question you are writing is:"},
    #     ]   
    #     stream = openai.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     stream=True
    #     )
    #     result = ""
    #     for chunk in stream:
    #         result += chunk.choices[0].delta.content or ""
    #         yield result 

    # def messages_for(self, question: str, model='gpt') :
    #     """
    #     Create the message list to be included in a call to OpenAI
    #     With the system and user prompt
    #     :param question: a question you want to ask
    #     :param related: related passages in the documents provided

    #     :return: the list of messages in the format expected by OpenAI
    #     """
    #     system_message = "You write a question for a math test. If you are not sure say you don't know."
    #     user_prompt = "The subject of the question is: "
    #     user_prompt += question
    #     user_prompt += "To make context for the question, here is a piece of text from a math textbook:"
    #     for related in self.find_related(question):
    #         user_prompt += related

    #     return self.chat_response(system_message, user_prompt)
    
    # def create_test(self, subject, grade_level, question_number, topics,question_type):
    #     system_message =f"You write a question for a {subject} test for grade level {grade_level}"
    #     user_prompt = f"Write {question_number} questions of type {question_type}. The topics are {topics}: "
    #     # for related in self.find_related(topics):
    #     #     user_prompt += related
    #     messages = [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_prompt},
    #         {"role": "assistant", "content": "The question you are writing is:"},
    #     ]
    #     stream = self.openai.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     stream=True
    #     )
    #     result = ""
    #     for chunk in stream:
    #         result += chunk.choices[0].delta.content or ""
    #         yield result 

    def create_formatted_test(self, subject, grade_level, question_number, topics, question_type, example_questions='', feedback='', creativity=0):
        all_related = ''
        system_message =f"You write questions for a {subject} test for grade level {grade_level}. Just respond with questions"
        user_prompt = f"Write {question_number} questions of type {question_type}. The topics are {topics}: "
        if self.collection:
            for related in self.find_related(topics):
                all_related += related
                user_prompt += all_related
        if feedback:
            user_prompt += feedback
        if example_questions:
            user_prompt += "Here are some example questions: {example_questions}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "The Questions are:"}
        ]
        completion = self.openai.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=messages,
        temperature = creativity/10,
        response_format = TestQuestions,
        )
        result = completion.choices[0].message.parsed
        self.DraftTest = result
        print(self.DraftTest)
        return result


    def stream_test(self):
        test_markdown = "## Draft Test Questions:\n\n\n"
        if type(self.DraftTest) == TestQuestions:
            tests = self.DraftTest.questions
            for item in tests:
            # test_markdown += item
                test_markdown += (f"{item.number}. {item.question} \n\n\n")
                yield test_markdown

    def restrict_questions_by(self, numbers):
        print('sefesefsgwg')
        restricted_questions = [question for question in self.DraftTest.questions if question.number in numbers]
        # self.DraftTest.questions = restricted_questions
        self.restricted_questions.extend(restricted_questions)

        self.FinalTest = TestQuestions(
        questions=[
        Question(number=i + 1, question=q.question) for i, q in enumerate(restricted_questions)
             ]
        )
        print(self.FinalTest.questions)
        return restricted_questions

    def create_test_file(self):
        return self.FinalTest.create_test_file()
    def find_related(self, description: str):
        """
        Return a list of text related to the given one by looking in the Chroma datastore
        """
        self.log("Agent is performing a RAG search of the Chroma datastore to find related text")
        vector = self.EMBEDDING_MODEL.encode(description)
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=3)
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
    
        