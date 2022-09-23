#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(URL, destination):
    URL = "https://drive.google.com/uc?id=1jQZfAMXiTbzlBKxI2PuJPP-zRaLJwlGV&export=download"

    session = requests.Session()

    response = session.get(URL, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
