import os
import base64
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

print("🔥 DEBUG GOOGLE_SYNC LOADED")

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

TOKEN_FILE = "token.pkl"
CREDENTIALS_FILE = "credentials.json"


def authenticate_gmail():
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)

    # 🔐 Print authenticated Gmail account
    profile = service.users().getProfile(userId='me').execute()
    print("\n🔐 Authenticated Gmail Account:", profile.get("emailAddress"))

    return service


def fetch_recent_emails(max_results=5):
    service = authenticate_gmail()

    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX'],
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    emails = []

    if not messages:
        print("No messages found.")
        return emails

    for msg in messages:
        message = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='full'
        ).execute()

        payload = message.get('payload', {})
        headers = payload.get('headers', [])

        subject = ""
        sender = ""

        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']

        body = ""

        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode()
                        break
        else:
            data = payload.get('body', {}).get('data')
            if data:
                body = base64.urlsafe_b64decode(data).decode()

        body = body.replace("\r", "").replace("\n", " ")

        print("\n📩 Fetched Email")
        print("From:", sender)
        print("Subject:", subject)

        email_text = f"Subject: {subject}\nFrom: {sender}\nSnippet: {body[:300]}"

        emails.append(email_text)

    return emails