import gzip
import shutil
from pathlib import Path

from utils import info, warning, fetch_file


def main():
    dataset_path = Path('../data/dataset')

    if not dataset_path.exists():
        dataset_path.mkdir()

    urls = ['https://datasets.imdbws.com/name.basics.tsv.gz',
            'https://datasets.imdbws.com/title.akas.tsv.gz',
            'https://datasets.imdbws.com/title.basics.tsv.gz',
            'https://datasets.imdbws.com/title.crew.tsv.gz',
            'https://datasets.imdbws.com/title.episode.tsv.gz',
            'https://datasets.imdbws.com/title.principals.tsv.gz',
            'https://datasets.imdbws.com/title.ratings.tsv.gz']

    destination_paths = [dataset_path / url[url.rfind('/') + 1:] for url in urls]

    for url, dest_path in zip(urls, destination_paths):
        if not dest_path.exists():
            info(f'Downloading {dest_path} from {url}')
            status = fetch_file(url, str(dest_path))
            if status != 200:
                warning(f'Downloading has failed with status {status}')

    unziped_paths = [dest_path.with_suffix('') for dest_path in destination_paths]

    for dest_path, unzipped_path in zip(destination_paths, unziped_paths):
        if not unzipped_path.exists():
            with gzip.open(dest_path, 'rb') as file_in:
                with open(unzipped_path, 'wb') as file_out:
                    info(f'Decompressing file {dest_path} into {unzipped_path}')
                    shutil.copyfileobj(file_in, file_out)


if __name__ == '__main__':
    main()
