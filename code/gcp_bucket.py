# Modeled after GCP documentation
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
 
def main():
    #for i in range(16507):
        #in_file =  'images/' + f'{i:05}' + '.png'
        #out_file = '/home/data/' + f'{i:05}' + '.png'
    download_blob('lsp_renum', 'joints.txt', 'lsp_joints')
    download_blob('mpii_renum', 'joints.txt', 'mpii_joints')
if __name__ == '__main__':
    main()
