# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:50:21 2025

@author: brian
"""
# Importing libraries
import imaplib
import email
from email.header import decode_header
import webbrowser
import os

from flask import Flask
from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
from dotenv import load_dotenv

import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px





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
                        # extract content type of email
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        try:
                            # get the email body
                            body = body + part.get_payload(decode=True).decode()
                        except:
                            pass
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            # print text/plain emails and skip attachments
                            # print("Body:", Body)
                            '''
                        elif "attachment" in content_disposition:
                            # download attachment
                            filename = part.get_filename()
                            if filename:
                                folder_name = clean(subject)
                                if not os.path.isdir(folder_name):
                                    # make a folder for this email (named after the subject)
                                    os.mkdir(folder_name)
                                filepath = os.path.join(folder_name, filename)
                                # download attachment and save it
                                open(filepath, "wb").write(part.get_payload(decode=True))
                        '''
                    Body.append(body)
                else:
                    # extract content type of email
                    content_type = msg.get_content_type()
                    # get the email body
                    Body.append(msg.get_payload(decode=True).decode())
                    if content_type == "text/plain":
                        # print only text email parts
                        # print(Body)
                        '''
                if content_type == "text/html":
                    # if it's HTML, create a new HTML file and open it in browser
                    folder_name = clean(subject)
                    if not os.path.isdir(folder_name):
                        # make a folder for this email (named after the subject)
                        os.mkdir(folder_name)
                    filename = "index.html"
                    filepath = os.path.join(folder_name, filename)
                    # write the file
                    open(filepath, "w").write(body)
                    # open in the default browser
                    webbrowser.open(filepath)
                '''
                # print("="*100)
                # print('\n\n')
    # close the connection and logout
    return Subject, From, Date, Body
    



# This loads the .env file
load_dotenv('ID.env')
load_dotenv('PASSWORD.env')
load_dotenv('SECRET_KEY.env')


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
con.login(os.getenv("ID")[:-1], os.getenv("PASSWORD")[:-1])

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
#---------------------------------------------------------------------------------------------------
# Create the layout of the app
#---------------------------------------------------------------------------------------------------
if status == 'OK':
    color = 'green'
else:
    color = 'red'
app.layout = html.Div([
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
                            'border':'1px solid rgba(0,0,0,.3)'},
                id='back'),
    html.Button('>', style={'display':'inline-block','padding-left':'3px','width':'30px',
                            'align-items': 'center', 'justify-content': 'center',
                            'border-radius':'5px',
                            'border':'1px solid rgba(0,0,0,.3)'},
                id='forward'),
    html.Br(),
    html.Div([html.Div([dcc.Markdown(f'**Subject**: {Subj}'), 
                        dcc.Markdown(f'**From**: {Fr}'), 
                        dcc.Markdown(f'**Date**: {Dat}'), 
                        #html.Br(),
                        ], style={'border':'1px solid rgba(0, 0, 0, .3)',
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
    html.Div(f'{Body[0]}',id='body', style={'display':'inline-block',
                                            'vertical-align': 'top',
                                            'padding-left':'30px',
                                            'overflow-y':'auto','overflow-wrap':'break-word',
                                            'width':'800px','height':'600px',},
             className="scrollbar_style")
    ], className='body')




@app.callback(
    Output('body', 'children', allow_duplicate=True),
    Output('email_0', 'n_clicks'),
    Input('email_0', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_0(n_click):
    return Body[0],0
    
@app.callback(
    Output('body', 'children', allow_duplicate=True),
    Output('email_1', 'n_clicks'),
    Input('email_1', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_1(n_click):
    return Body[1],0
    
    
@app.callback(
    Output('body', 'children', allow_duplicate=True),
    Output('email_2', 'n_clicks'),
    Input('email_2', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_2(n_click):
    return Body[2],0
    
    
@app.callback(
    Output('body', 'children', allow_duplicate=True),
    Output('email_3', 'n_clicks'),
    Input('email_3', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_3(n_click):
    return Body[3],0
    
    
@app.callback(
    Output('body', 'children', allow_duplicate=True),
    Output('email_4', 'n_clicks'),
    Input('email_4', 'n_clicks'),
    prevent_initial_call=True,
)
    
def Click_for_body_4(n_click):
    return Body[4],0
    
@app.callback(
    Output('body','children', allow_duplicate=True),
    Output('email_range','children', allow_duplicate=True),
    Output('email_0','children', allow_duplicate=True),
    Output('email_1','children', allow_duplicate=True),
    Output('email_2','children', allow_duplicate=True),
    Output('email_3','children', allow_duplicate=True),
    Output('email_4','children', allow_duplicate=True),
    Input('back', 'n_clicks'),
    State('email_range','children'),
    prevent_initial_call=True,
)
    
def back_range(n_clicks,email_range):
    email_start, email_end = int(email_range.split(' - ')[0]), int(email_range.split(' - ')[1])
    
    if email_start == 0 or email_end <= 5:
        return email_range
    else:
        email_start -= 5
        email_end -= 5
        global Body
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
        
        return Body[0], str(email_start) + ' - ' + str(email_end), grouped_Div[0], grouped_Div[1], grouped_Div[2], grouped_Div[3], grouped_Div[4]

    
@app.callback(
    Output('body','children', allow_duplicate=True),
    Output('email_range','children', allow_duplicate=True),
    Output('email_0','children', allow_duplicate=True),
    Output('email_1','children', allow_duplicate=True),
    Output('email_2','children', allow_duplicate=True),
    Output('email_3','children', allow_duplicate=True),
    Output('email_4','children', allow_duplicate=True),
    Input('forward', 'n_clicks'),
    State('email_range','children'),
    prevent_initial_call=True,
)
    
def forward_range(n_clicks,email_range):
    email_start, email_end = int(email_range.split(' - ')[0]), int(email_range.split(' - ')[1])
    
    if email_end == messages or email_end >= messages-5:
        return email_range
    else:
        email_start += 5
        email_end += 5
        global Body
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
        
        return Body[0], str(email_start) + ' - ' + str(email_end), grouped_Div[0], grouped_Div[1], grouped_Div[2], grouped_Div[3], grouped_Div[4]
    
    