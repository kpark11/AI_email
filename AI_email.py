# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:50:21 2025

@author: brian
"""
# Importing libraries
import imaplib, email
 

def get_credential():
    with open("credentials.txt", 'r') as file:
        parts = file.read()
    credential = parts.split(",")
    return credential
        
 
# Function to get email content part i.e its body part
def get_body(msg):
    if msg.is_multipart():
        return get_body(msg.get_payload(0))
    else:
        return msg.get_payload(None, True)
 
# Function to search for a key value pair 
def search(key, value, con): 
    result, data = con.search(None, key, '"{}"'.format(value))
    return data
 
# Function to get the list of emails under this label
def get_emails(result_bytes, con):
    msgs = [] # all the email data are pushed inside an array
    for num in result_bytes[0].split():
        typ, data = con.fetch(num, '(RFC822)')
        msgs.append(data[0][1].decode('utf-8'))
    return msgs
 

def main():
    
    user, password = get_credential()
    
    imap_url = 'imap.gmail.com'
    imap_port = 993
            
    # this is done to make SSL connection with GMAIL
    con = imaplib.IMAP4_SSL(imap_url, imap_port) 
    
    # logging the user in
    con.login(user, password)
    
    # calling function to check for email under this label
    con.select('Inbox') 
    
    # fetching emails from this user "tu**h*****1@gmail.com"
    global msgs
    msgs = get_emails(search('FROM', 'kimanpark33@gmail.com', con), con)
    
    # Uncomment this to see what actually comes as data 
    # print(msgs) 
    
    # Finding the required content from our msgs
    # User can make custom changes in this part to
    # fetch the required content he / she needs
     
    # printing them by the order they are displayed in your gmail 
    for msg in msgs[-5:-1]: 
        for sent in msg:
            if type(sent) is tuple: 
     
                # encoding set as utf-8
                content = str(sent[1], 'utf-8')
                data = str(content)
     
                # Handling errors related to unicodenecode
                try: 
                    indexstart = data.find("ltr")
                    data2 = data[indexstart + 5: len(data)]
                    indexend = data2.find("</div>")
     
                    # printing the required content which we need
                    # to extract from our email i.e our body
                    print(data2[0: indexend])
     
                except UnicodeEncodeError as e:
                    pass
    
    
    
    
    

if __name__ == "__main__":
    main()
    
