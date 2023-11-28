import os
import argparse
import json
import requests
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_mesh_tags(article):
    """
    Parse an article xml to extract all MeSH tags and Publication Types

    Parameters
    ----------
    article: xml.etree.ElementTree.Element
        xml of the article

    Returns
    -------
    list
        list of MeSH tags
    list
        list of Publication Types
    """
    pm_id = next(article.iter("PMID")).text
    meshs = [mesh[0].text for mesh in article.iter("MeshHeading")]
    publication_types = [pt.text for pt in article.iter("PublicationType")]

    return pm_id, meshs, publication_types


def update_metadata_mesh(records, pm_ids):
    """
    Update records by fetching their MeSH tags and Publication Tags, using the 
    EFetch API of the Entrez E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch)

    Parameters
    ----------
    records: dict
        dictionary of PubMed articles, with associated information and metadata
    pm_ids: list
        list of PubMed ids
    """
    api_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pm_ids}
    max_retry = 10
    retry = 0
    while(True):
        try: 
            response = requests.post(api_url, params=params)
            root = ET.fromstring(response.content)
            new_records = []
            for article_xml in root:
                pm_id, meshs, publication_types = get_mesh_tags(article_xml)
                for record in records:
                    if record['externalids']['pubmed'] == pm_id:
                        record.update({"mesh": meshs, "publicationtype": publication_types})
                        new_records += [record]
            break
        except Exception as oops:
            if oops is StopIteration:
                print(response.url)
            retry += 1
            if retry >= max_retry:
                raise RuntimeError("Error communicating with the server: %s" % oops)
            print(f"Error communicating with server ({retry}):", oops)
            time.sleep(1)

    return new_records



def augment_PMC(source_path, dest_path, log_path, batch_size=100):
    print('='*100+f'\n4. Scraping MeSH and Publication Type tags for PMC articles in {source_path}.\n')

    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('0')

    with open(log_path, 'r') as f_in:
        start_line = int(f_in.read())

    # aligning the two files, to start from the first article not yet processed
    with open(source_path, 'r') as f_in:
        for _ in tqdm(range(start_line), desc="Skipping already processed articles..."):
            next(f_in)     
        
        with open(dest_path, 'a') as f_out:
            count = 0
            records, pm_ids = [], []
            for i, line in tqdm(enumerate(f_in), initial=start_line, total=4900000, desc="Processing new articles..."):
                record = json.loads(line)
                pm_id = record['externalids']['pubmed']
                if pm_id is None:
                    continue
                records += [record]
                pm_ids += [pm_id]
                count += 1
                
                # Scrape metadata in batches
                if count % batch_size == 0:
                    new_records = update_metadata_mesh(records, pm_ids)
                    for new_record in new_records:
                        if new_record:
                            f_out.write(json.dumps(new_record) + '\n')
                    records, pm_ids = [], []

                    with open(log_path, 'w') as f_log:
                        f_log.write(str(start_line+i+1))

            # last batch
            if count % batch_size > 0:
                new_records = update_metadata_mesh(records, pm_ids)
                for new_record in new_records:
                    f_out.write(json.dumps(new_record) + '\n')
    print(f"Finished scraping MeSH metadata for {count} articles.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Inpute PubMedCentral file, after the first metadata scraping step (i.e. after running PMC_load.py).")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output PubMedCentral file, enriched of MeSH tags and Publication Types.")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Textual file to store the last processed line."
    )
    args = parser.parse_args()
    print(args)

    augment_PMC(args.input_path, args.output_path, log_path=args.log_path)

if __name__ == "__main__":
    main()