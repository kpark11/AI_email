# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:50:21 2025

@author: brian
"""
# Importing libraries
import imaplib
import email
from email.header import decode_header
import os

from flask import Flask
from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
from dotenv import load_dotenv

import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px

from bs4 import BeautifulSoup

# For Sentiment Analysis
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

from transformers import pipeline

import re
from spellchecker import SpellChecker
from langdetect import detect
from num2words import num2words
import emoji
from nltk import word_tokenize
from unidecode import unidecode
import torch
import transformers
from huggingface_hub import login


def getEmail(con, start, end):
    Subject = []
    From = []
    Date = []
    Body = []
    for i in range(start, end, -1):
        # fetch the email message by ID
        res, msg = con.fetch(str(i), "(RFC822)")
        for response in msg:
            if isinstance(response, tuple):
                # parse a bytes email into a message object
                msg = email.message_from_bytes(response[1])
                
                # decode the email subject
                Subj, encoding = decode_header(msg["Subject"])[0]
                if isinstance(Subj, bytes):
                    # if it's a bytes, decode to str
                    if encoding is None:
                        Subject.append(Subj.decode('utf-8'))
                    else:
                        Subject.append(Subj.decode(encoding))
                else:
                    Subject.append(Subj)
                
                # decode email sender
                Fr, encoding = decode_header(msg.get("From"))[0]
                if isinstance(Fr, bytes):
                    if encoding is None:
                        From.append(Fr.decode('utf-8'))
                    else:
                        From.append(Fr.decode(encoding))
                else:
                    From.append(Fr)
                    
                # decode email date
                Dat, encoding = decode_header(msg.get("Date"))[0]
                if isinstance(Dat, bytes):
                    if encoding is None:
                        Date.append(Dat.code('utf-8'))
                    else:
                        Date.append(Dat.decode(encoding))
                else:
                    Date.append(Dat)
                
                # print("Subject:", Subject)
                # print("From:", From)
                # print("Date:", Date)
                # print("\n")
                body = ''
                # if the email message is multipart
                if msg.is_multipart():
                    # iterate over email parts
                    for part in msg.walk():
                        
                        # Check if there is text/html
                        content_types = []
                        for part2 in msg.walk():
                            content_type = part2.get_content_type()
                            content_types.append(content_type)
                            
                        
                        # extract content type of email
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        if 'text/html' in content_types:
                            if content_type == 'text/html':
                                try:
                                    html = part.get_payload(decode=True).decode()
                                except:
                                    html = part.get_payload(decode=True).decode('unicode_escape')
                                soup = BeautifulSoup(html, 'html.parser')
                                body += soup.get_text()
                                Body.append(body)
                                break
                            else:
                                pass
                        
                        elif content_type == "text/plain":
                            try:
                                html = part.get_payload(decode=True).decode()
                            except:
                                html = part.get_payload(decode=True).decode('unicode_escape')
                            soup = BeautifulSoup(html, 'html.parser')
                            body += soup.get_text()
                            Body.append(body)
                            break
                else:
                    content_type = msg.get_content_type()
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        body += msg.get_payload(decode=True).decode()
                        Body.append(body)
                    elif content_type == "text/html":
                        try:
                            html = msg.get_payload(decode=True).decode()
                        except:
                            html = msg.get_payload(decode=True).decode('unicode_escape')
                        soup = BeautifulSoup(html, 'html.parser')
                        body += soup.get_text()
                        Body.append(body)
                    else:
                        body += msg.get_payload(decode=True).decode()
                        Body.append(body)
                # print("="*100)
                # print('\n\n')
    # close the connection and logout
    return Subject, From, Date, Body
    



# This loads the .env file
load_dotenv('.env')





def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def remove_special_characters(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

def remove_urls(text):
    # Define a regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Use the re.sub() function to replace all occurrences of URLs with an empty string
    clean_text = re.sub(url_pattern, '', text)
    return clean_text

def convert_numbers_to_words(text):
    words = []
    for word in text.split():
        if word.isnumeric():
            words.append(num2words(word))
        else:
            words.append(word)
    return ' '.join(words)

def convert_accented_to_ascii(text):
    return unidecode(text)

def convert_emojis_to_words(text):
    text = emoji.demojize(text)
    return text

def convert_to_lowercase(text):
    lowercased_text = text.lower()
    return lowercased_text

def remove_whitespace(text):
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def truncate(text):
    tokens = word_tokenize(text)
    tokens = tokens[:512]
    text = ' '.join(tokens)
    return text

'''
def correct_spelling(text):
    spell = SpellChecker()
    tokens = word_tokenize(text)
    print('tokens: ', tokens)
    corrected_tokens = [spell.correction(word) if word != None else '' for word in tokens]
    print('corrected_tokens: ', corrected_tokens)
    corrected_text = ' '.join(corrected_tokens)
    print('corrected_text: ', corrected_text)
    return corrected_text
'''

def detect_language(text):
    try:
        language = detect(text)
    except:
        language = 'unknown'
    return language


#os.environ["HF_HUB_ETAG_TIMEOUT"] = 600
# Load models for sentiment analysis, summarizer, and text generation

# Sentiment Analysis
sentiment_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=sentiment_MODEL, tokenizer=sentiment_MODEL)

def sentiment(text):
    #print('*'*20)
    #print('sentiment: ')

    #print(text)
    result = sentiment_task(text)
    #print(result)
    return f'{result[0]["label"]} ({result[0]["score"]})'
    
    

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summary(text):
    #print('*'*20)
    #print('summary: ')
    text = remove_html_tags(text)
    text = convert_emojis_to_words(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = convert_numbers_to_words(text)
    text = convert_accented_to_ascii(text)
    text = convert_to_lowercase(text)
    text = remove_whitespace(text)
    text = truncate(text)
    #text = correct_spelling(text)
    #print(text)
    token_num = len([word for word in text.split(' ')])
    summary = summarizer(text, max_length=token_num, min_length=5, do_sample=False)
    #print(summary[0]['summary_text'])
    return summary[0]['summary_text']

# Have to login in with "huggingface-cli login"
# Automatic reply text generation
login(token=os.getenv('HF_READ'))
gen_model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=gen_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    
)

def auto_reply(text):
    
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always \
         responds in pirate speak! You are currently reply to a given email."},
        {"role": "user", "content": f"{text}"},
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=400,
    )
    print(outputs)
    return outputs[0]["generated_text"][-1]

#---------------------------------------------------------------------------------------------------
# Initialize the Flask and Dash app
#---------------------------------------------------------------------------------------------------


# Exposing the Flask Server to enable configuring it for logging in
server = Flask(__name__)
app = dash.Dash(
    __name__, server=server, use_pages=True, suppress_callback_exceptions=True
)

# Define imap url and port
imap_url = 'imap.gmail.com'
imap_port = 993

# this is done to make SSL connection with GMAIL
con = imaplib.IMAP4_SSL(imap_url, imap_port) 

# logging the user in
con.login(os.getenv("ID"), os.getenv("PASSWORD"))

# Updating the Flask Server configuration with Secret Key to encrypt the user session cookie
server.config.update(SECRET_KEY=os.getenv("SECRET_KEY"))

#---------------------------------------------------------------------------------------------------
# Set the title of the dashboard
#---------------------------------------------------------------------------------------------------
app.title = "AI email"
app.style = {'textAlign':'center','color':'#000000','font-size':24}

app.run_server(debug=True)




# calling function to check for email under this label
status, selections = con.list()
#print('Status:', status)
#print('\n')
#print('selections:')
#for selection in selections:
#    print(selection.decode("utf-8").split('"/"'))
status, messages = con.select('Inbox') 
#print('\n')

# total number of emails
messages = int(messages[0])
#print('number of messages: ' + str(messages) + '\n\n')

start_n = 0
end_n = 5

Subject, From, Date, Body = getEmail(con, messages, messages-5)
summary_text = summary(Body[0])
sentiment_text = sentiment(summary_text)
sent_color = {'positive':'green', 'neutral':'yellow', 'negative':'red'}
#---------------------------------------------------------------------------------------------------
# Create the layout of the app
#---------------------------------------------------------------------------------------------------
if status == 'OK':
    color = 'green'
else:
    color = 'red'
app.layout = html.Div([
    html.H1('AI Email', className='header'),
    html.H1(f'Status: {status}', style={'color':color}),
    #html.Br(),
    html.Label('Selections:'),
    dcc.Dropdown(
        id='selection',
        options=[{'label': selection.decode("utf-8").split('"/"')[-1], 
                  'value': selection.decode("utf-8").split('"/"')[-1]} for selection in selections],
        value='INBOX',
        #placeholder='Selection',
        style={'width':'200px'}
        ),
    html.Br(),
    html.Div(f'Total number of emails: {str(messages)}', style={'display':'inline-block'}),
    html.Div(f'{start_n} - {end_n}', style={'display':'inline-block',
                                            'padding-left':'80px','padding-right':'20px'},
             id='email_range'),
    html.Button('<', style={'display':'inline-block','padding-left':'3px','width':'30px',
                            'align-items': 'center', 'justify-content': 'center',
                            'border-radius':'5px',
                            'background':'linear-gradient(to right, #76b852 0%, #8DC26F  51%, #76b852  100%)',
                            #'border':'1px solid rgba(0,0,0,.3)',
                            },
                id='back'),
    html.Button('>', style={'display':'inline-block','padding-left':'3px','width':'30px',
                            'align-items': 'center', 'justify-content': 'center',
                            'border-radius':'5px', 
                            'background':'linear-gradient(to right, #76b852 0%, #8DC26F  51%, #76b852  100%)',
                            #'border':'1px solid rgba(0,0,0,.3)',
                            },
                id='forward'),
    
         
    html.Br(),
    html.Div([html.Div([dcc.Markdown(f'**Subject**: {Subj}'), 
                        dcc.Markdown(f'**From**: {Fr}'), 
                        dcc.Markdown(f'**Date**: {Dat}'), 
                        #html.Br(),
                        ], style={'border':'1px solid rgba(225, 225, 225, .3)',
                                  'width':'450px',
                                  #'text-align':'center',
                                  'padding-left':'5px',
                                  'padding-top':'5px',
                                  'border-radius':'10px',
                                  'cursor': 'pointer',
                                  },
                        id=f'email_{i}',
                        className='list') 
              for i, (Subj,Fr,Dat) in enumerate(zip(Subject, From, Date))], style={'display':'inline-block'}),
    html.Div(dcc.Loading([
                    dcc.Markdown(f'**Subject**: {Subject[0]}'),
                    dcc.Markdown(f'**From**: {From[0]}'),
                    dcc.Markdown(f'**Date**: {Date[0]}'),

                    html.Div(f'{Body[0]}',id='body', style={'overflow-y':'auto','overflow-wrap':'break-word',},
                             className="scrollbar_style")], 
        type='circle', id='loading'), style={'display':'inline-block',
                                                     'vertical-align': 'top',
                                                     'padding-left':'30px',
                                                     'overflow-y':'auto','overflow-wrap':'break-word',
                                                     'width':'600px','height':'600px',}, className="scrollbar_style"),
    html.Div([html.Div([
        dcc.Loading(dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'}), id='sentiment-analysis'),
        html.Br(),
        html.Br(),
        dcc.Loading(dcc.Markdown(f'**Summary**: {summary_text}'), id='summary'),
        html.Br(),
        html.Br(),
        html.Button('Automatic Reply', id='text_generation', className='Button'),
        dcc.Loading('', id='reply')
             ])], style={'width':'200px', 'display':'inline-block', 'vertical-align':'top',
                         'padding-left':'100px'}),
    

                                                                                
    ], className='body')




@app.callback(
    Output('loading', 'children', allow_duplicate=True),
    Output('email_0', 'n_clicks', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('email_0', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_0(n_click):
    Subj_choice = dcc.Markdown(f'**Subject**: {Subject[0]}') 
    Fr_choice = dcc.Markdown(f'**From**: {From[0]}') 
    Dat_choice = dcc.Markdown(f'**Date**: {Date[0]}')
    summary_text = summary(Body[0])
    summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
    sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
    
    return [Subj_choice, Fr_choice, Dat_choice, Body[0]], 0, sentiment_txt, summary_txt
    
@app.callback(
    Output('loading', 'children', allow_duplicate=True),
    Output('email_1', 'n_clicks', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('email_1', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_1(n_click):
    Subj_choice = dcc.Markdown(f'**Subject**: {Subject[1]}') 
    Fr_choice = dcc.Markdown(f'**From**: {From[1]}') 
    Dat_choice = dcc.Markdown(f'**Date**: {Date[1]}')
    summary_text = summary(Body[1])
    summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
    sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
    return [Subj_choice, Fr_choice, Dat_choice, Body[1]], 0, sentiment_txt, summary_txt
    
    
@app.callback(
    Output('loading', 'children', allow_duplicate=True),
    Output('email_2', 'n_clicks', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('email_2', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_2(n_click):
    Subj_choice = dcc.Markdown(f'**Subject**: {Subject[2]}') 
    Fr_choice = dcc.Markdown(f'**From**: {From[2]}') 
    Dat_choice = dcc.Markdown(f'**Date**: {Date[2]}')
    summary_text = summary(Body[2])
    summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
    sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
    return [Subj_choice, Fr_choice, Dat_choice, Body[2]], 0, sentiment_txt, summary_txt
    
    
@app.callback(
    Output('loading', 'children', allow_duplicate=True),
    Output('email_3', 'n_clicks', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('email_3', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_3(n_click):
    Subj_choice = dcc.Markdown(f'**Subject**: {Subject[3]}') 
    Fr_choice = dcc.Markdown(f'**From**: {From[3]}') 
    Dat_choice = dcc.Markdown(f'**Date**: {Date[3]}')
    summary_text = summary(Body[3])
    summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
    sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
    return [Subj_choice, Fr_choice, Dat_choice, Body[3]], 0, sentiment_txt, summary_txt
    
    
@app.callback(
    Output('loading', 'children', allow_duplicate=True),
    Output('email_4', 'n_clicks',    allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('email_4', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_4(n_click):
    Subj_choice = dcc.Markdown(f'**Subject**: {Subject[4]}') 
    Fr_choice = dcc.Markdown(f'**From**: {From[4]}') 
    Dat_choice = dcc.Markdown(f'**Date**: {Date[4]}')
    summary_text = summary(Body[4])
    summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
    sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
    return [Subj_choice, Fr_choice, Dat_choice, Body[4]], 0, sentiment_txt, summary_txt
    
@app.callback(
    Output('loading','children', allow_duplicate=True),
    Output('email_range','children', allow_duplicate=True),
    Output('email_0','children', allow_duplicate=True),
    Output('email_1','children', allow_duplicate=True),
    Output('email_2','children', allow_duplicate=True),
    Output('email_3','children', allow_duplicate=True),
    Output('email_4','children', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('back', 'n_clicks'),
    State('email_range','children'),
    prevent_initial_call=True,
)
    
def back_range(n_clicks,email_range):
    email_start, email_end = int(email_range.split(' - ')[0]), int(email_range.split(' - ')[1])
    if email_start <= 0 or email_end <= 5:
        raise PreventUpdate
    else:
        email_start -= 5
        email_end -= 5
        global Subject, From, Date, Body
        Subject, From, Date, Body = getEmail(con, messages-email_start, messages-email_end)
        
        grouped_Div = [html.Div([dcc.Markdown(f'**Subject**: {Subj}'), 
                            dcc.Markdown(f'**From**: {Fr}'), 
                            dcc.Markdown(f'**Date**: {Dat}'), 
                            #html.Br(),
                            ], #style={'border':'1px solid rgba(0, 0, 0, .3)',
                               #       'width':'400px',
                               #       #'text-align':'center',
                               #       'padding-left':'5px',
                               #       'padding-top':'5px',
                               #       'cursor': 'pointer',
                               #       },
                            id=f'email_{i}') for i, (Subj,Fr,Dat) in enumerate(zip(Subject, From, Date))]

        Subj_choice = dcc.Markdown(f'**Subject**: {Subject[0]}') 
        Fr_choice = dcc.Markdown(f'**From**: {From[0]}') 
        Dat_choice = dcc.Markdown(f'**Date**: {Date[0]}')
        
        summary_text = summary(Body[0])
        sentiment_text = sentiment(summary_text)
        summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
        sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})        
        
        return [Subj_choice, Fr_choice, Dat_choice, Body[0]], \
    str(email_start) + ' - ' + str(email_end), grouped_Div[0], \
    grouped_Div[1], grouped_Div[2], grouped_Div[3], grouped_Div[4], \
    sentiment_txt, summary_txt

@app.callback(
    Output('loading','children', allow_duplicate=True),
    Output('email_range','children', allow_duplicate=True),
    Output('email_0','children', allow_duplicate=True),
    Output('email_1','children', allow_duplicate=True),
    Output('email_2','children', allow_duplicate=True),
    Output('email_3','children', allow_duplicate=True),
    Output('email_4','children', allow_duplicate=True),
    Output('sentiment-analysis','children', allow_duplicate=True),
    Output('summary','children', allow_duplicate=True),
    Input('forward', 'n_clicks'),
    State('email_range','children'),
    prevent_initial_call=True,
)
    
def forward_range(n_clicks,email_range):
    email_start, email_end = int(email_range.split(' - ')[0]), int(email_range.split(' - ')[1])
    
    if email_end == messages or email_end >= messages-5:
        raise PreventUpdate
    else:
        email_start += 5
        email_end += 5
        global Subject, From, Date, Body
        Subject, From, Date, Body = getEmail(con, messages-email_start, messages-email_end)
        
        grouped_Div = [html.Div([dcc.Markdown(f'**Subject**: {Subj}'), 
                            dcc.Markdown(f'**From**: {Fr}'), 
                            dcc.Markdown(f'**Date**: {Dat}'), 
                            #html.Br(),
                            ], #style={'border':'1px solid rgba(0, 0, 0, .3)',
                               #       'width':'400px',
                               #       #'text-align':'center',
                               #       'padding-left':'5px',
                               #       'padding-top':'5px',
                               #       'cursor': 'pointer',
                               #       },
                            id=f'email_{i}') for i, (Subj,Fr,Dat) in enumerate(zip(Subject, From, Date))]
        
        Subj_choice = dcc.Markdown(f'**Subject**: {Subject[0]}')
        Fr_choice = dcc.Markdown(f'**From**: {From[0]}')
        Dat_choice = dcc.Markdown(f'**Date**: {Date[0]}')
        
        summary_text = summary(Body[0])
        sentiment_text = sentiment(summary_text)
        summary_txt = dcc.Markdown(f'**Summary**: {summary_text}')
        sentiment_txt = dcc.Markdown(f'**Sentiment Analysis**: {sentiment_text}', style={'color':f'{sent_color[sentiment_text.split(" ")[0]]}'})
        
        return [Subj_choice, Fr_choice, Dat_choice, Body[0]], \
    str(email_start) + ' - ' + str(email_end), grouped_Div[0], \
    grouped_Div[1], grouped_Div[2], grouped_Div[3], grouped_Div[4], \
    sentiment_txt, summary_txt
    


@app.callback(
    Output('reply','children'),
    Input('text_generation','n_clicks'),
    State('body','children')
)

def generate_text(n_clicks,body):
    text = auto_reply(body)
    return text
