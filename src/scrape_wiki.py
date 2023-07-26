import pickle
import re
from pathlib import Path

import hydra
import mwparserfromhell as parser
import pandas as pd
import pywikibot
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import info, warning, error, log_into_wandb


def scrape(page: pywikibot.Page, log_unparsed: bool = True) -> (str, dict):
    def log_header(page, node_n=None):
        node_info = f' node# {node_n}' if node_n is not None else ''
        header = f'Parsing {page} id={page.pageid} {node_info}: '
        return header

    def fix_movie_title(movie_title: str) -> str:
        """
        Cleans up a movie title as produced by mwparserfromhell and then returns it
        :param movie_title: the movie title to be cleaned up.
        :return: the movie title as a human would expect it
        """
        if movie_title[:5] != '[[en:':
            info(f'Found movie title that doesn begin by `[[en:`: {movie_title}')
        else:
            movie_title = movie_title[5:]
        if movie_title[-2:] != ']]':
            info(f'Found movie title that doesn end by `]]`: {movie_title}')
        else:
            movie_title = movie_title[:-2]
        pattern = r'\(.*?film.*?\)'
        match = re.search(pattern, movie_title)
        if match:
            substring = match.group()
            movie_title = movie_title.replace(substring, '')
        movie_title = movie_title.strip()
        return movie_title

    def fix_title(properties: dict, page_title: str) -> str:
        display_title = properties.get('displaytitle')
        if display_title:
            prefix_pos = display_title.find('<i>')
            if prefix_pos >= 0:
                display_title = display_title[prefix_pos + 3:]
            suffix_pos = display_title.find('</i>')
            if suffix_pos >= 0:
                display_title = display_title[:suffix_pos]
            display_title = display_title.strip()
            return display_title
        res = fix_movie_title(page_title)
        return res

    def get_metadata_from_infobox(params: list[parser.nodes.extras.parameter.Parameter]) -> dict:
        title = None
        year_found = None
        for i, param in enumerate(params):
            if not isinstance(param, parser.nodes.extras.parameter.Parameter):
                warning(f'Found node# {i} in infobox which is not a Parameter: {str(param)}')
                continue
            if param.name.strip() == 'name':
                title = str(param.value).rstrip('\n').strip()
            elif param.name.strip() == 'released':
                param_value = str(param.value)
                years = ['2021', '2022', '2023', '2024', '2025', '2026']
                for year in years:
                    if year in param_value:
                        year_found = int(year)
                        break

        return {'title': title, 'year': year_found}

    page_text = page.text
    wikicode = parser.parse(page_text)
    # Fine the beginning and the end of the Plot, or Synopsis, section
    # compiled_re_strict = re.compile('^\s*(Plot|Synopsis)\s*$')
    compiled_re = re.compile('^.*(plot|synopsis).*$', flags=re.IGNORECASE)
    plot_beginning = None
    plot_end = None
    title_from_infobox = ''
    release_year = None
    for i, node in enumerate(wikicode.nodes):
        # If the beginning of the Plot section hasn't been found yet, then look for it
        if isinstance(node, parser.nodes.template.Template) and node.name.rstrip(' \n') == 'Infobox film':
            res = get_metadata_from_infobox(node.params)
            title_from_infobox = res['title']
            #  if not title_from_infobox:
            #      info(f"{log_header(page, i)}couldn't parse title from infobox")
            release_year = res['year']
            # if not release_year:
            #     info(f"{log_header(page, i)}couldn't parse release year from infobox")

        if plot_beginning is None and isinstance(node, parser.nodes.heading.Heading):
            node_title = str(node.title)
            if not compiled_re.match(node_title):
                continue
            #  if not compiled_re_strict.match(node_title):
            #      info(f'{log_header(page, i)}Header title `{node_title}` matches Plot/Synopsis but not strictly')
            plot_beginning = i + 1  # Exclude the Plot/Synopsis header itself from the section
            continue
        # If the beginning of the Plot section has been found already, then find where it ends, by looking for the
        # beginning of the next section
        if isinstance(node, parser.nodes.heading.Heading):
            plot_end = i  # The beginning of the next section, one position after the actual end of the Plot,
            break

    if plot_beginning is None or plot_end is None:
        if plot_beginning is None and log_unparsed:
            info(f"{log_header(page)}could not parse as didn't find the beginning of a Plot/Synopsis section")
        elif plot_end is None and log_unparsed:
            info(f"{log_header(page)}could not parse as didn't find the end of a Plot/Synopsis section")
        return '', None
    # Extract the plot as a string.
    # Start by making a list of strings that, once concatenated, will produce the full text
    plot_strings = []
    for pos, node in enumerate(wikicode.nodes[plot_beginning: plot_end]):
        match type(node):
            case parser.nodes.text.Text:
                plot_strings.append(node.value)
            case parser.nodes.Wikilink:
                if isinstance(node.text, parser.wikicode.Wikicode) and str(node.text):
                    # There could be a [thumb] here, skip it, e.g. see pageid 69971492
                    if str(node.text)[:6] != 'thumb|':
                        plot_strings.append(str(node.text))
                    continue
                plot_strings.append(str(node.text) if node.text else str(node.title))
                if not node.text and not node.title:
                    warning(
                        f'{log_header(page, pos + plot_beginning)}found Wikilink node with title and text both empty.')
            case parser.nodes.Comment:
                continue
            case parser.nodes.Tag:
                continue
            case parser.nodes.Template:
                continue
            case parser.nodes.html_entity.HTMLEntity:
                continue
            case parser.nodes.ExternalLink:
                plot_strings.append(str(node.title))
            case _:
                warning(
                    f'{log_header(page, pos + plot_beginning)}found unexpected and unhandled node type {type(node)}.')

    res = ''.join(plot_strings)
    res = res.lstrip(' \n').rstrip(' \n')
    if not res:
        # info(f"{log_header(page)}Plot/Synopsis section found but it is empty")
        return '', None
    fixed_title = fix_title(page.properties(), str(page))
    title = fixed_title

    metadata = {'title': title,
                'year': int(release_year) if release_year else -1,
                'id': page.pageid,
                'revision_id': page.latest_revision_id}  # This is NOT the same as a page id

    return res, metadata


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project
    params = params.scrape_wiki
    config_path = Path('../config')
    load_dotenv(config_path / '.env')

    log_into_wandb()
    data_path = Path('../data')
    if not data_path.exists():
        data_path.mkdir()

    not_parsed_list_path = data_path / params.failed_parsing_list
    dataset_pickle_path = data_path / params.pickled_dataset

    # Create a site object for the English Wikipedia
    site = pywikibot.Site("en", "wikipedia")
    pywikibot_basedir = pywikibot.config.get_base_dir()
    info(f'Base directory for pywikibot: {pywikibot_basedir}')

    wiki_pages_2021 = list(pywikibot.Category(site, '2021_films').articles())
    wiki_pages_2022 = list(pywikibot.Category(site, '2022_films').articles())
    wiki_pages_2023 = list(pywikibot.Category(site, '2023_films').articles())
    wiki_pages = [*wiki_pages_2021, *wiki_pages_2022, *wiki_pages_2023]
    wiki_pages = wiki_pages[:20]

    scrape_config = params.get('scrape')
    if (isinstance(scrape_config, str) and scrape_config == 'all') or not scrape_config:
        to_be_scraped = wiki_pages
    else:
        requested_ids = set(params.scrape)
        to_be_scraped = [page for page in wiki_pages if page.pageid in requested_ids]
        if len(to_be_scraped) < len(requested_ids):
            warning(f'Configuration parameters listed {len(requested_ids)} IDs for pages to be scraped, \
            but only {len(to_be_scraped)} of those could be found in the searched Wikipedia categories')

    all_metadata = []
    parsed_successfully = 0
    pages_not_parsed = []
    dataset = {'data': {}, 'metadata': None}
    found_ids = set()
    for page in tqdm(to_be_scraped):
        scraped, metadata = scrape(page, log_unparsed=False)
        if metadata is None:
            url = f'https://en.wikipedia.org/?curid={page.pageid}'
            pages_not_parsed.append(url + '\n')
            continue
        if metadata['id'] in found_ids:
            info(f"Found page with duplicate id {metadata['id']}, metadata={metadata}")
            continue
        found_ids.add(metadata['id'])
        all_metadata.append(metadata)
        dataset['data'][int(page.pageid)] = scraped
        parsed_successfully += 1

    info(f'Parsed successfully the information for {parsed_successfully} movie(s) out of {len(to_be_scraped)}')

    with wandb.init(project=wandb_project,
                    notes="Makes a dataset with 2021-'22-'23 movies info scraping it from Wikipedia",
                    config={'params': OmegaConf.to_object(params)}):

        info(f"Saving the list of ids for pages that couldn't be parsed into {not_parsed_list_path}")
        with open(not_parsed_list_path, 'wt') as not_parsed_f:
            not_parsed_f.writelines(pages_not_parsed)

        # Pickle and save the dataset, parsed text along with its metadata
        all_metadata_df = pd.DataFrame(all_metadata)
        all_metadata_df.set_index('id', inplace=True, drop=False)
        dataset['metadata'] = all_metadata_df
        # Check that there are no duplicate page ids

        unique_ids_count = all_metadata_df.id.nunique()
        if unique_ids_count != len(all_metadata_df):
            error(f'Found {len(all_metadata_df) - unique_ids_count} non-unique page ids')
        info(f'Saving dataset with its metadata into {dataset_pickle_path}')
        with open(dataset_pickle_path, 'bw') as dataset_f:
            pickle.dump(dataset, dataset_f, protocol=pickle.HIGHEST_PROTOCOL)
        dataset_artifact = wandb.Artifact(name='movies_dataset',
                                          type='dataset',
                                          description='Pickled Python data structure with the dataset and its metadata')
        dataset_artifact.add_file(dataset_pickle_path)
        dataset_artifact.add_file(not_parsed_list_path)
        wandb.log_artifact(dataset_artifact)


if __name__ == '__main__':
    main()

"""

=============>> Use page.properties to find info about the title

Check pageid 73928775, 74338121, 74334236 -> Done
Problem with title of 69336552, 65266767 -> Done
Check this out, 'thumb|Historical photo [...]' pageid  69971492 -> Done
Consolidate text and metadata into one dict and then pickle it to save it on disk -> Done
Relocate the apicache-py3 under data -> Not gonna happen
Clean-up the title and add the year to metadata -> Done
Reject plots that are a string of blanks and \n only, e.g. for movie id 72923309 -> Done

See what other metadata you want to fetch and save, e.g. movie director(s) and genre
Some categories of films don't have a plot, e.g. documentaries. Also, TV shows have a plot formatted in a different way
Should I handle those indicating them as such in the logged message?
"""
