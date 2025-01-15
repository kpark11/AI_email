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
 


def get_credential():
    with open("credentials.txt", 'r') as file:
        parts = file.read()
    credential = parts.split(",")
    return credential
        

def main():
    
    user, password = get_credential()
    
    imap_url = 'imap.gmail.com'
    imap_port = 993
            
    # this is done to make SSL connection with GMAIL
    con = imaplib.IMAP4_SSL(imap_url, imap_port) 
    
    # logging the user in
    con.login(user, password)
    
    # calling function to check for email under this label
    selection = con.list()
    print('selections: ' + str(selection) + '\n\n')
    status, messages = con.select('Inbox') 
    
    # number of top emails to fetch
    N = 3
    # total number of emails
    messages = int(messages[0])
    print('number of messages: ' + str(messages) + '\n\n')
    
    for i in range(messages, messages-N, -1):
        # fetch the email message by ID
        res, msg = con.fetch(str(i), "(RFC822)")
        for response in msg:
            if isinstance(response, tuple):
                # parse a bytes email into a message object
                msg = email.message_from_bytes(response[1])
                
                # decode the email subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    # if it's a bytes, decode to str
                    subject = subject.decode(encoding)
                
                # decode email sender
                From, encoding = decode_header(msg.get("From"))[0]
                if isinstance(From, bytes):
                    From = From.decode(encoding)
                
                print("Subject:", subject)
                print("From:", From)
                
                # if the email message is multipart
                if msg.is_multipart():
                    # iterate over email parts
                    for part in msg.walk():
                        # extract content type of email
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        try:
                            # get the email body
                            body = part.get_payload(decode=True).decode()
                        except:
                            pass
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            # print text/plain emails and skip attachments
                            print(body)
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
                else:
                    # extract content type of email
                    content_type = msg.get_content_type()
                    # get the email body
                    body = msg.get_payload(decode=True).decode()
                    if content_type == "text/plain":
                        # print only text email parts
                        print(body)
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
                print("="*100)
    # close the connection and logout
    con.close()
    con.logout()
    
    

if __name__ == "__main__":
    main()
    
