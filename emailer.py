import smtplib, ssl
    

def send_email_using_gmail(emailId, password, to, subject, body):
    context = ssl.create_default_context()
    print(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        try:
            server.login(emailId, password)
            server.sendmail(emailId, to, f"Subject: {subject}\n{body}")
            return True
        except:
            return False