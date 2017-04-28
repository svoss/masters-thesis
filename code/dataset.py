from config import get_config


def connect_to_stack():
    config = get_config()
    options = {
         'webdav_hostname': config.get('stack','host'),
         'webdav_login': config.get('stack', 'username'),
         'webdav_password':config.get('stack', 'password'),
         'verbose':True
    }

    import webdav.client as wc
    client = wc.Client(options)
    return client


def extract_zip_into(from_file,to):
    import zipfile
    with zipfile.ZipFile(from_file, "r") as z:
        z.extractall(to)


def download_file_from_url(url, to_path):
    import requests
    r = requests.get(url, stream=True)
    with open(to_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    return to_path


if __name__ == '__main__':

    print extract_zip_into('/Users/stijnvoss/Desktop/test.zip','/Users/stijnvoss/Desktop/test/')