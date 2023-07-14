from pathlib import Path

import pandas as pd


def compile_metadata(starting_year, ending_year, dataset_path, output_path):

    title_basics = pd.read_table(dataset_path / 'title.basics.tsv', low_memory=False)
    metadata_titles = pd.DataFrame()
    metadata_titles['title'] = title_basics.primaryTitle
    metadata_titles['year'] = pd.to_numeric(title_basics.startYear, errors='coerce')
    metadata_titles['title_type'] = title_basics.titleType
    metadata_titles['title_id'] = title_basics.tconst
    metadata_titles = metadata_titles[
        metadata_titles.year.notna() & (metadata_titles.year >= starting_year) & (
                metadata_titles.year <= ending_year) & (
                metadata_titles.title_type == 'movie')]
    metadata_titles.year = metadata_titles.year.astype(int)
    del title_basics

    title_crew = pd.read_table(dataset_path / 'title.crew.tsv')
    metadata_crew = pd.DataFrame()
    metadata_crew['title_id'] = title_crew.tconst
    metadata_crew['director_ids'] = title_crew.directors
    metadata_crew.set_index('title_id', inplace=True, verify_integrity=True)
    del title_crew

    name_basics = pd.read_table(dataset_path / 'name.basics.tsv')
    metadata_names = pd.DataFrame()
    metadata_names['name_id'] = name_basics.nconst
    metadata_names['name'] = name_basics.primaryName
    metadata_names.set_index('name_id', inplace=True, verify_integrity=True)
    del name_basics

    def title_id_to_directors(title_id: str) -> str:
        director_ids = metadata_crew.director_ids[title_id].split(',')
        director_names = [metadata_names.name[name_id] for name_id in director_ids if name_id != '\\N']
        res = ','.join(director_names)
        return res

    metadata_titles['directors'] = metadata_titles.title_id.apply(title_id_to_directors)
    metadata_titles.set_index('title_id', inplace= True, verify_integrity= True)
    metadata_titles.sort_index(inplace=True)
    metadata_titles.to_csv(output_path)


def main():
    starting_year = 2022
    ending_year = 2023

    dataset_path = Path('../data/dataset')
    output_path = Path('../data/metadata.csv')

    compile_metadata(starting_year, ending_year, dataset_path, output_path)


if __name__ == '__main__':
    main()
