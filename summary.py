# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:59:50 2022

@author: damini
"""

# summary 


from transformers import pipeline
    
def summary(whole_text):
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(whole_text, max_length=130, min_length=30, do_sample=False)