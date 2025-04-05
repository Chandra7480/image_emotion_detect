import gdown

# Google Drive file ID
file_id = '1YhUd9YCfuJHIe6cPWUhO8rEdgYxk6aH7'

# Create the download link
url = f'https://drive.google.com/uc?id={file_id}'

# Output filename
output = 'emotion_detection_model.pkl'

# Download the file
gdown.download(url, output, quiet=False,fuzzy=True)

