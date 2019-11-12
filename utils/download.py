import requests
from tqdm import tqdm
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
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
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_model():
    if os.path.exists('tmp_/pytorch_model.bin') and os.path.exists('tmp_/config.json'):
        return
    else:
        os.mkdir('tmp_')
        file_id = '1vp7Gs0O_XThfji8Df6Z_VKPHYQYr4mPp'
        destination = 'tmp_/pytorch_model.bin'
        download_file_from_google_drive(file_id, destination)
        file_id = '1CZciWt9BdYA6WlDyvtcSQpEhYEYGFD_H'
        destination = 'tmp_/config.json'
        download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
    file_id = '1CZciWt9BdYA6WlDyvtcSQpEhYEYGFD_H'
    destination = 'config.json'
    download_file_from_google_drive(file_id, destination)
# https://drive.google.com/file/d/1vp7Gs0O_XThfji8Df6Z_VKPHYQYr4mPp/view?usp=sharing

# https://drive.google.com/file/d/1CZciWt9BdYA6WlDyvtcSQpEhYEYGFD_H/view?usp=sharing