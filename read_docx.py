import docx
import json
import re

doc = docx.Document("Neural Network Laboratory Record Updated.docx")
fullText = []
for para in doc.paragraphs:
    fullText.append(para.text)

content = "\n".join(fullText)
print(content)
