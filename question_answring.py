# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 01:14:56 2022

@author: damini
"""

# question answring


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import textwrap


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
input_ids = tokenizer.encode(question, answer_text)
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    
    if id == tokenizer.sep_token_id:
        print('')
    
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')

sep_index = input_ids.index(tokenizer.sep_token_id)

num_seg_a = sep_index + 1

num_seg_b = len(input_ids) - num_seg_a

segment_ids = [0]*num_seg_a + [1]*num_seg_b

assert len(segment_ids) == len(input_ids)

outputs = model(torch.tensor([input_ids]),
                             token_type_ids=torch.tensor([segment_ids]), 
                             return_dict=True) 

start_scores = outputs.start_logits
end_scores = outputs.end_logits

answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')

answer = tokens[answer_start]

for i in range(answer_start + 1, answer_end + 1):
    
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''

    input_ids = tokenizer.encode(question, answer_text)

    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)
    
    outputs = model(torch.tensor([input_ids]), 
                    token_type_ids=torch.tensor([segment_ids]), 
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    
wrapper = textwrap.TextWrapper(width=80) 

test_corpus = '''swedish cars have a strong reputation for vehicle and passenger safety. Not only that, the entire country is considered a model of road traffic safety worldwide. Statistically, the year 2015 saw only 2.4 traffic deaths per 100,000 inhabitants in Sweden, whereas the global average was 17.4 deaths, with 10.4 in the U.S. and still 4.3 in Germany. Sweden’s strong safety record can be traced back to its politicians, who paved the way for improved safety early on. In 1997, the Swedish Parliament laid the foundation for Vision Zero in its transport policy. The goal was to reduce to zero the number of deaths and severe injuries resulting from traffic accidents. Other countries – Germany among them – followed Sweden’s lead as did organizations like the European Union. By the year 2050, the EU plans to reach almost “zero loss of life” through traffic accidents.
 '''

print(wrapper.fill(test_corpus))


question = "what is the goal by 2050"
answer_question(question, test_corpus)


