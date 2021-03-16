import os
import glob
import email

path = 'data/'

ham_path = glob.glob(path+'ham/easy_ham/*')
ham_2_path = glob.glob(path+'ham/easy_ham_2/*')
hard_ham_path = glob.glob(path+'ham/hard_ham/*')
ham_paths = [
    ham_path,
    ham_2_path,
    hard_ham_path
]

spam_path = glob.glob(path+'spam/spam/*')
spam_2_path = glob.glob(path+'spam/spam_2/*')
spam_paths = [
    spam_path,
    spam_2_path
]

def get_email_content(email_path):
    file = open(email_path, encoding='latin1')
    try:
        msg = email.message_from_file(file)
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload()
    except Exception as e:
        print(e)

def get_email_content_bulk(email_paths):
    email_contents = [get_email_content(o) for o in email_paths]
    return email_contents
