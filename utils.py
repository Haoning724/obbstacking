import pickle
import os
import zipfile


def pickle_dump(obj, url):
    pickle.dump(obj, open(url, 'wb'))


def pickle_load(url):
    return pickle.load(open(url, 'rb'))


def create_zip_file(output_dir, input_dir, file_name="test.zip"):
    output_zip = os.path.join(output_dir, file_name)
    if os.path.isfile(output_zip):
        os.remove(output_zip)

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))

    zipf = zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED)
    zipdir(input_dir, zipf)
    zipf.close()
