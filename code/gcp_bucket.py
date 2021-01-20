# Modeled after GCP documentation
from google.cloud import storage
import os
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
    argv = ['S7', 'Discussion 1.54138969', 5415]
    folder = argv[0] + '_cropped/' + argv[1]

    if not os.path.isdir('/home/gregory_barboy/data/' + argv[0] + '_cropped/'):
        os.mkdir('/home/gregory_barboy/data/' + argv[0] + '_cropped/')
    if not os.path.isdir('/home/gregory_barboy/data/' + folder + '/'):
        print(folder + '/')
        os.mkdir('/home/gregory_barboy/data/' + folder)
    num_images = argv[2]
    # for i in range(num_images):
    #     in_file = folder + '/' + f'{i:05}' + '.png'
    #     out_file = '/home/gregory_barboy/data/' + folder + '/' + f'{i:05}' + '.png'
    #     download_blob('h36m_processed', in_file, out_file)
    download_blob('h36m_processed', folder + '/' + 'joints.txt', folder + '/' + 'joints.txt')
if __name__ == '__main__':
    main()
