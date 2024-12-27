from pydantic import BaseModel
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import datetime


class Question(BaseModel):
    number: int
    question: str

class TestQuestions(BaseModel):
    questions : List[Question]
    # format : str
    # icon: image
    # def __init__(self, question_list):
    #     self.questions = [question_list]
    #     self.format = 'pdf'

    def keep_questions(self, numbers):
        self.questions = [question for question in self.questions if question.number in numbers]

    def stream_questions(self):
        for question in self.questions:
            yield f"{question.int}. {question}"
    
    def create_test_file(self, test_name='', test_date='', type='pdf',icon_path=None):
        filename = 'Test-{date:%Y-%m-%d_%H:%M:%S}.pdf'.format( date=datetime.datetime.now() )   
        # text = ''.join([f"{item.number}. {item.question} \n\n" for item in result.questions])
        # file_path = "output.pdf"
        # if type == 'pdf':
        #      write_text_to_pdf(text, file_path)
        # else:
        #     with open(file_path, "w") as f:
        #             f.write(text)
        #     # f.close()
        # return file_path, file_path
    
        # Initialize PDF
        pdf = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Set margins
        margin = 50

        # Draw icon
        if icon_path:
            pdf.drawImage(icon_path, margin, height - 100, width=50, height=50)

        # Test title
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(margin + 60, height - 70, test_name)

        # Date
        pdf.setFont("Helvetica", 12)
        pdf.drawString(margin, height - 120, f"Date: {test_date}")

        # Name field
        pdf.drawString(margin, height - 140, "Name: _____________________________")

        # Line separator
        pdf.setLineWidth(1)
        pdf.setStrokeColor(colors.black)
        pdf.line(margin, height - 160, width - margin, height - 160)

        # Questions
        pdf.setFont("Helvetica", 12)
        y_position = height - 200
        question_number = 1

        for question in self.questions:
            pdf.drawString(margin, y_position, f"{question_number}. {question}")
            y_position -= 40  # Adjust spacing between questions

            # Add a new page if the questions exceed the page limit
            if y_position < margin:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = height - 50
            question_number += 1

        # Save the PDF
        pdf.save()
        return filename




    
    

    
        