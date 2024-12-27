from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    number: int
    question: str

class TestQuestions(BaseModel):
    questions : List[Question]
    # format : str

    def stream_questions(self):
        for question in self.questions:
            yield f"{question.int}. {question}"
    

    
    

    
        