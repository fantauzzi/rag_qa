import pickle
import re
from pathlib import Path

import mwparserfromhell as parser
import pandas as pd
import pywikibot
from tqdm import tqdm

from utils import info, warning, error


def scrape(page: pywikibot.Page, log_unparsed: bool = True) -> (str, dict):
    def log_header(page, node_n=None):
        node_info = f' node# {node_n}' if node_n else ''
        header = f'Parsing {page} id={page.pageid} {node_info}: '
        return header

    page_text = page.text
    wikicode = parser.parse(page_text)
    # Fine the beginning and the end of the Plot, or Synopsis, section
    compiled_re_strict = re.compile('^\s*(Plot|Synopsis)\s*$')
    compiled_re = re.compile('^.*(plot|synopsis).*$', flags=re.IGNORECASE)
    plot_beginning = None
    plot_end = None
    for i, node in enumerate(wikicode.nodes):
        # If the beginning of the Plot section hasn't been found yet, then look for it
        if plot_beginning is None and isinstance(node, parser.nodes.heading.Heading):
            node_title = str(node.title)
            if not compiled_re.match(node_title):
                continue
            if not compiled_re_strict.match(node_title):
                info(f'{log_header(page, i)}Header title `{node_title}` matches Plot/Synopsis but not strictly')
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
                # There could be a [thumb] here, skip it, e.g. see pageid 69971492
                if isinstance(node.text, parser.wikicode.Wikicode):
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
    res = res.lstrip('\n').rstrip('\n')
    if not res:
        info(f"{log_header(page)}Plot/Synopsis section found but it is empty")
        return '', None

    metadata = {'title': str(page),
                'id': page.pageid,
                'revision_id': page.latest_revision_id}  # This is NOT the same as a page id
    return res, metadata


def main() -> None:
    dataset_path = Path('../data/dataset')
    if not dataset_path.exists():
        dataset_path.mkdir()
    # Create a site object for the English Wikipedia
    site = pywikibot.Site("en", "wikipedia")

    wiki_pages_2021 = list(pywikibot.Category(site, '2021_films').articles())
    wiki_pages_2022 = list(pywikibot.Category(site, '2022_films').articles())
    wiki_pages_2023 = list(pywikibot.Category(site, '2023_films').articles())
    wiki_pages = [*wiki_pages_2021, *wiki_pages_2022, *wiki_pages_2023]
    # wiki_pages2 = [page for page in wiki_pages if page.pageid == 69971492]
    all_metadata = []
    parsed_successfully = 0
    pages_not_parsed = []
    dataset = {'data': {}, 'metadata': None}
    found_ids = set()
    for page in tqdm(wiki_pages):
        scraped, metadata = scrape(page, log_unparsed=False)
        if metadata is None:
            url = f'https://en.wikipedia.org/?curid={page.pageid}'
            pages_not_parsed.append(url + '\n')
            continue
        if metadata['id'] in found_ids:
            info(f"Found page with duplicate id {metadata['id']}, metadata={metadata}")
            url = f'https://en.wikipedia.org/?curid={page.pageid}'
            pages_not_parsed.append(url + '\n')
            continue
        found_ids.add(metadata['id'])
        all_metadata.append(metadata)
        dataset['data'][int(page.pageid)] = scraped
        parsed_successfully += 1

    info(f'Parsed successfully the information for {parsed_successfully} movie(s) out of {len(wiki_pages)}')
    not_parsed_list_path = dataset_path / 'not_parsed.txt'

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
    # all_metadata_df.to_csv(dataset_path / 'metadata.csv')
    dataset_pickle_path = dataset_path / 'dataset.pickle'
    info(f'Saving dataset with its metadata into {dataset_pickle_path}')
    with open(dataset_pickle_path, 'bw') as dataset_f:
        pickle.dump(dataset, dataset_f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

"""
Relocate the apicache-py3 under data

Check this out, 'thumb|Historical photo [...]' pageid  69971492 -> Done
Consolidate text and metadata into one dict and then pickle it to save it on disk -> Done
Reject plots that are a string of blanks and \n only, e.g. for movie id 72923309
See what other metadata you want to fetch and save, e.g. movie director(s) and genre
Some categories of films don't have a plot, e.g. documentaries. Also, TV shows have a plot formatted in a different way
Should I handle those indicating them as such in the logged message?

pattern = r'\(.*?(2021|2022|2023).*?\)'
text = "Howdy!"
if re.search(pattern, text):
    print("Found!")
"""
