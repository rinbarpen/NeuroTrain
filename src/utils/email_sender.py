# TODO: Not implementation!

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import logging
from pathlib import Path

class EmailSender:
    def __init__(self, system_email: str, system_email_password: str, receiver_email: str):
        self.sender_email = system_email
        self.sender_password = system_email_password
        self.receiver_email = receiver_email
        self.subject = ""
        self.body = ""
        self.smtp_url = "smtp.gmail.com"
        self.smtp_port = 587
        
        self.smtp_server = smtplib.SMTP(self.smtp_url, self.smtp_port)
        self.smtp_server.login(self.sender_email, self.sender_password)

    def send(self, attachment_path: Path|None=None):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = self.subject

        msg.attach(MIMEText(self.body, 'plain'))

        if attachment_path is not None and attachment_path.exists():
            with attachment_path.open("rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename= {attachment_path.name}")
                msg.attach(part)

        try:
            self.smtp_server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            logging.info("Email sent successfully!")
        except Exception as e:
            logging.warning(f"Error sending email: {e}")
