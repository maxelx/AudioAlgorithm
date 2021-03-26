import requests
#URL = "http://127.0.0.1:5000/predict"
URL = "https://score-rest.herokuapp.com/predict"
TEST_AUDIO_FILE = "AudioFIle/voce.wav"

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE, "rb")
    values = {"file" : (TEST_AUDIO_FILE, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    print(response)
    data = response.json()

    print("Predicted measure are : " + str(data['Result']))