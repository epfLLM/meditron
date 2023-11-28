from lxml import etree
import requests
import argparse
import json
import os
from tqdm import tqdm

BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&field=title&term=\"{}\"'

def extract_ref_author(ref_text):
    author = ref_text.split('.')[0]
    if ',' in author:
        author = author.split(', ')[0]
    return author

def extract_ref_year(ref_text):
    year = ref_text
    year = year[:year.rfind(';')]
    year = year[year.rfind(' ')+1:]
    return year

def extract_ref_title(ref_text): 
    ref_title = ref_text
    if ref_title[-1] == '.':
        ref_title = ref_title[:-1]
    ref_title = ref_title[ref_title.find('.')+2:]
    ref_title = ref_title[:1+ref_title.rfind('.')]
    return ref_title

def request_IDs(query):
    ''' Fetch list of PubMed IDs for a given e-search query. '''
    try: 
        url_request = BASE_URL.format(query)
        page = requests.get(url_request).content
        content = etree.fromstring(page)
        root = content.getroottree()
        result = [item.text for item in root.xpath("//Id")]
        return result
    except: 
        return []

def scrape_references(ref_path, ids_path, start, end, batch_size=5, verbose=True):
    # Read pre-scraped pubmed IDs
    with open(ids_path, 'r') as f_in:
        scraped_ids = set([line.strip() for line in f_in])
        print(f'Loaded {len(scraped_ids)} scraped PubMed IDs.')

    print(f'Scraping PubMed IDs for references {start} to {end} by batches of {batch_size}...')
    with open(ref_path, 'r') as f_in, open(ids_path, 'a') as f_out:
        batch = []
        batch_idx = start // batch_size
        num_matches = 0
        for i, line in enumerate(f_in):
            # Scrape IDs for references between start and end indices
            if i < start:
                continue
            if end is not None and i >= end:
                break

            # Load batch of batch_size Uptodate references
            try: 
                ref_text = json.loads(line)['ref_text']
                if ref_text == '': 
                    continue
                ref_title = extract_ref_title(ref_text)
                ref_year = extract_ref_year(ref_text)
                ref_author = extract_ref_author(ref_text)
                if ref_title == '' or ref_year == '' or ref_author == '':
                    continue
                query = f'(({ref_author} [Author - First]) AND ({ref_title} [Title]) AND ({ref_year} [Date - Publication]))'
            except: 
                print(f'Batch {batch_idx}: Error loading reference: {line}')
                continue

            # Continue until you fill up the batch
            batch.append(query)
            if len(batch) < batch_size: 
                continue

            # Request PubMed IDs for batch using first author, paper title and pub date
            batch_idx += 1
            batch_query = ' OR '.join(batch)
            matching_ids = request_IDs(batch_query)
            num_found = len(matching_ids)
            num_matches += num_found
            matching_ids = [id for id in matching_ids if id not in scraped_ids]
            scraped_ids = scraped_ids.union(set(matching_ids))
            if verbose: 
                print(f'Batch {batch_idx}: Adding {len(matching_ids)} of {num_found} found PubMed IDs: {matching_ids}')
            if len(matching_ids) > 0: 
                f_out.write('\n'.join(matching_ids) + '\n')
            batch = []
    if end: 
        print(f'Finished scraping PubMed IDs for references {start} to {end}.')
        print(f'Found PubMed IDs for {num_matches} out of {end-start} articles.')


def identify_references(ids_path, data_path, source):
    ''' Identify PubMed articles referenced in Uptodate.'''

    # Scrape Uptodate PubMed IDs 
    pubmed_ids = set()

    if source == "uptodate":
        with open(ids_path, 'r') as f_in:
            for line in f_in.readlines():
                pubmed_ids.add(line.strip())
    elif source == "cochrane":
        with open(ids_path, 'r') as f_in:
            data = json.load(f_in)
        pubmed_ids = {article["DOI"].lower() for article in data}
    else:
        raise ValueError("Invalid source")
    
    # Add uptodate_reference field to pubmed articles referenced in Uptodate
    out_path = data_path.split('.')[0]+f'_{source}.jsonl'
    key = source + '_reference'
    with open(data_path, 'r') as f_in, open(out_path, 'w') as f_out:
        # Read line by line the data path (streaming because it's a big file)
        for line in tqdm(f_in, total=4700000):
            article = json.loads(line)
            article[key] = 0
            try:
                if source == "uptodate":
                    pm_id = article['externalids']['pubmed']
                elif source == "cochrane":
                    pm_id = article['externalids']['doi'].lower()
                else:
                    raise ValueError("Invalid source")
                if pm_id in pubmed_ids:
                    article[key] = 1
            except:
                pass
            f_out.write(json.dumps(article) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refs_path",
        type=str,
        required=False,
        help="Uptodate references file, without PubMedIDs (for mode 'scrape').")
    parser.add_argument(
        "--ids_path",
        type=str,
        required=True,
        help="PubMed IDs of Uptodate References.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Pubmed articles file (for mode 'identify').")
    parser.add_argument(
        "--start",
        type=int,
        required=False,
        default=0,
        help="Start index of references to divide.")
    parser.add_argument(
        "--end",
        type=int,
        required=False,
        default=None,
        help="End index of references to divide.")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size for E-Utils API calls.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode: 'scrape' or 'identify'.")
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        help="Source: 'cochrane' or 'uptodate'.")
    args = parser.parse_args()
    print(args)

    # identify: Add identifier to pubmed articles referenced in Uptodate
    if args.mode == 'identify':
        identify_references(args.ids_path, args.data_path, args.source)

    # scrape: Scrape PubMed IDs for Uptodate references
    elif args.mode == 'scrape':
        scrape_references(
            args.refs_path, args.ids_path, args.start, args.end, 
            batch_size=args.batch_size, verbose=True)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')

if __name__ == "__main__":
    main()

# python3 reference.py --ids_path /pure-mlo-scratch/data/pubmed/uptodate_pubmed_ids.jsonl --data_path /pure-mlo-scratch/data/pubmed/pubmed_processed_mesh_train.jsonl --mode identify
