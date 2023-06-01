from langchain import PromptTemplate
import langchain
import pandas as pd
import os
from langchain import HuggingFaceHub, LLMChain
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

context = "Aboobaker is father of Latheef and Mymoonath is mother"
question = "Who is the mother of latheef?"

encoding = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])

print("Question:", question)
print("Answer:", answer)


template = """Question: {question}


Context:{context}


Answer: """
question = "Who is the father of latheef?"
context="Aboobaker is father of Latheef"
prompt = PromptTemplate(
        template=template,
    input_variables=['question','context']
)
# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)
# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)
# ask the user question about NFL 2010
print(llm_chain.run(question))
filename='land'
file_path='C:/Users/lathe/PROJECT/DB/'
df = pd.read_csv(file_path+filename+'.csv')


