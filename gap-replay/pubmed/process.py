"""
This script implements the pre-processing of PubMed full-text articles and abstracts. 
Including: Filtering, formatting, cleaning. 
"""

import argparse
import json
import numpy as np
import tqdm.auto as tqdm
import json
from itertools import groupby
import re
import os
from langdetect import detect
import jsonlines

from load import *

KEEP_HEADER = False         # Keep article header (content before title/abstract/first section header)?
KEEP_FIGURE_CONTENT = True  # Keep figure content and wrap in [fig] tokens?
KEEP_TABLE_CONTENT = True   # Keep table content and wrap in [table] tokens? 
KEEP_BIBLIOGRAPHY = False   # Keep bibligraphy entries and wrap in [bib] tokens?

SPECIAL_TOKENS = [
    '[bib_ref]', '[/bib_ref]',  # In-text author references
    '[fig_ref]', '[/fig_ref]',  # In-text figure references
    '[formula]', '[/formula]'   # In-text formulae
    ]
if KEEP_FIGURE_CONTENT:
    SPECIAL_TOKENS += ['[fig]', '[/fig]']
if KEEP_TABLE_CONTENT:
    SPECIAL_TOKENS += ['[table]', '[/table]']
if KEEP_BIBLIOGRAPHY: 
    SPECIAL_TOKENS += ['[bib]', '[/bib]']

MAIN_SECTION_HEADERS = [
    'Abstract', 'Introduction', 'Background', 'Related',
    'Method', 'Material', 'Result', 'Analysis', 'Discussion',
    'Conclusion', 'Contribution', 'Statement', 'Declaration', 
    'Strength', 'Limitation', 'Future research', 'Funding',
    'Disclosure', 'Acknowledgment', 'Ethical', 
    'Tables', 'Figures', 'Appendix'
]

def detect_lang(text, sample_size=2000): 
    '''
    Helper: Detect language of a given text.
    '''
    try:
        sample = text if len(text) < sample_size else text[:sample_size]
        language = detect(sample)
    except:
        language = 'unknown'
    return language

def remove_urls(text):
    '''
    Helper: remove URLs from text.
    '''
    return re.sub(
        r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', 
        text, flags=re.MULTILINE)

def remove_references(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    return text

def summarize_caption(caption, max_length):
    ''' 
    Helper: summarize figure caption to max_length words.  
    '''
    # Truncate first sentence if > 20 characters
    if len(caption) > 20:
        caption = re.split(r'[.;:()]', caption)[0]
    # Truncate to max_length words if needed
    split = caption.split()
    if len(split) > max_length:
        caption = ' '.join(split[:max_length])+'...'
    return caption

def is_main_section_header(section):
    '''
    Helper: check if a section header is a usual main section header.
    '''
    if len(section.split(' ')) > 3: 
        return False
    for header in MAIN_SECTION_HEADERS:
        if header.lower() in section.lower():
            return True
    return False

def format_bib(record, bib_id, max_length=12):
    ''' 
    Format in-text bibliography reference into (paper title, main author last name).
    Truncates bibliography title to max_length words if needed. 
    '''
    article = record['content']['text']
    annotations = record['content']['annotations']
    try:
        # Find bib entry 
        for bib_entry in json.loads(annotations['bibentry']):
            if bib_entry['attributes']['id'] == bib_id:
                entry_start = int(bib_entry['start'])
                entry_end = int(bib_entry['end'])
                break

        # Find title 
        for bib_title in json.loads(annotations['bibtitle']):
            if bib_title['start'] >= entry_start and bib_title['end'] <= entry_end:
                bib_title_str = article[int(bib_title['start']):int(bib_title['end'])]
                break

        # If no title found, skip reference
        if not bib_title_str:
            return None

        # Find main author's last name
        for bib_author in json.loads(annotations['bibauthorlastname']):
            if bib_author['start'] >= entry_start and bib_author['end'] <= entry_end:
                bib_author_name = article[int(bib_author['start']):int(bib_author['end'])]
                break
        if not bib_author_name:
            return None
    except:
        return None

    # Format bibliography reference
    split = bib_title_str.split()
    if len(split) > max_length:
        bib_title_str = ' '.join(split[:max_length])+'...'
    bib_str = f"{bib_title_str}, {bib_author_name}"
    return bib_str


def format_fig(record, fig_id, max_length=12):
    '''
    Format figure reference into `Fig [ID]: [summarized figure caption].`
    Truncates figure caption to max_length words if needed. 
    '''
    article = record['content']['text']
    annotations = record['content']['annotations']
    try:
        # Find figure entry
        for fig in json.loads(annotations['figure']):
            if fig['attributes']['id'] == fig_id:
                fig_start = int(fig['start'])
                fig_end = int(fig['end'])
                break


        # Find figure caption
        fig_caption_start, fig_caption_end = None, None
        for fig_caption in json.loads(annotations['figurecaption']):
            if fig_caption['start'] >= fig_start and fig_caption['end'] <= fig_end:
                fig_caption_start = int(fig_caption['start'])
                fig_caption_end = int(fig_caption['end'])
                break

        # If no caption found, skip
        if not fig_caption_start:
            return None, None

        # Format prefix 
        prefix = article[fig_start:fig_caption_start].split('\n')[1]
        fig_name = re.sub(r'[:()]', '', prefix.replace(' .', ' '))
        fig_name = fig_name.replace('Fig.', 'Figure')
        fig_name = fig_name.replace('Tab.', 'Table')
        fig_name = fig_name.replace(' Figure', ', Figure')
        fig_name = fig_name.strip()

        # Format caption
        caption = article[fig_caption_start:fig_caption_end].replace(prefix, '').strip()
        if max_length:
            caption = summarize_caption(caption, max_length)
        if caption.split()[0].isdigit():
            fig_name += ' '+caption.split()[0]
            caption = ' '.join(caption.split()[1:])
        if fig_name != '':
            fig_name += ': '
        while caption.startswith('.') or caption.startswith(',') or caption.startswith(')'):
            caption = caption[1:].strip()
        return fig_name, caption
    except:
        return None

def parse_article(record):
    '''
    Creates an array of annotation types for each character in the article.
    This array is then used to format the article using the `format_article` function.
    '''
    article = record['content']['text']
    if not article:
        return None
    reflect_array = np.array(['T' for _ in range(len(article))], dtype=object)
    parsing_dict = {
        'authorfirstname': 'RM',
        'authorlastname': 'RM',
        'authoraffiliation': 'RM',
        'bibentry': 'BIB',
        'formula': 'FML',
        'sectionheader': 'SEC',
        'bibref': None,
        'figureref': None,
        'tableref': None,
        'figure': None
    }

    # Parse each annotation type
    for annot_type, token in parsing_dict.items():
        annotations = record['content']['annotations'][annot_type]
        if not annotations:
            continue
        annotations = json.loads(annotations)

        # Remove title duplicates
        if annot_type == 'title':
            annotations = [annotations[0]]

        for annotation in annotations:
            start = int(annotation["start"])
            end = int(annotation["end"])
            try:
                # In-text references (skip unidentified ones!)
                if annot_type in ['bibref', 'figureref', 'tableref']:

                    # Fix recurrent parsing error
                    if '(' in article[start-3:start]:
                        start = start-3+article[start-3:start].index('(')
                    if ')' in article[end:end+3]:
                        end = end+article[end:end+3].index(')')+1

                    if 'attributes' in annotation.keys():
                        ref_id = annotation['attributes']['ref_id']
                        reflect_array[start:end] = ref_id
                    else: 
                        reflect_array[start:end] = 'b?'
                elif annot_type == 'figure':
                    fig_id = annotation['attributes']['id']
                    fig_id = fig_id.split('_')[0].upper()+'_'+fig_id.split('_')[1] 
                    reflect_array[start:end] = fig_id
                else:
                    reflect_array[start:end] = token
            except:
                pass
    
    # Remove article header (before title/abstract/first section header)
    if not KEEP_HEADER: 
        try:
            start = None
            abstract = record['content']['annotations']['abstract']
            if abstract:
                abstract_start = int(json.loads(abstract)[0]['start'])
                if abstract_start:
                    start = abstract_start
            section_headers = json.loads(record['content']['annotations']['sectionheader'])
            if section_headers:
                intro_start = min([int(s['start']) for s in section_headers])
                if not start or intro_start < start:
                    start = intro_start
            if start:
                idx_T = np.where(reflect_array == 'T')[0]
                idx_before_abstract = idx_T[idx_T < start]
                reflect_array[idx_before_abstract] = 'P'
        except:
            pass
    return reflect_array

def format_article(record):
    '''
    Full-text article formatting using S2ORC annotations.
    '''
    start = 0
    text = ''
    formatted_figs = {}
    formatted_bibs = {}
    added_figures = []
    article = record['content']['text']

    # Parse article into array of annotation types
    reflect_array = parse_article(record)

    # Group sections by annotation type
    split_array = [list(group) for _, group in groupby(reflect_array)]
    at_figures = False
    for subarray in split_array:
        end = start + len(subarray)
        annot_type = subarray[0]
        part = article[start:end]

        # Format whitespace and bullet points
        part = part.strip()
        part = part.replace('•', '- ')
        try: 

            # Skip empty sections 
            if part == '':
                start += len(subarray)
                continue 

            # Keep abstract & main body (skip all text after figures)
            elif annot_type == 'T' and not at_figures:
                text += part
            
            # Format section headers (# for sections, ## for subsections, capitalise first letter)
            elif annot_type == 'SEC':
                part = part[0].upper() + part[1:].lower()
                if is_main_section_header(part):
                    text += '\n# ' + part + '\n'
                else: 
                    text += '\n## ' + part + '\n'

            # Wrap entries in special tokens [bib] (only if KEEP_BIBLIOGRAPHY)
            elif annot_type == 'BIB' and KEEP_BIBLIOGRAPHY: 
                text += ' [bib] ' + part + ' [/bib]\n'

            # Wrap in-text figures/table refs in [fig_ref] tokens + summarize caption
            elif 'fig_' in annot_type or 'tab_' in annot_type:
                if annot_type in formatted_figs:
                    fig_str = formatted_figs[annot_type]
                else:
                    fig_name, caption = format_fig(record, annot_type)
                    fig_str = fig_name + caption
                    formatted_figs[annot_type] = fig_str
                if fig_str:
                    text += ' [fig_ref] ' + fig_str + ' [/fig_ref] '

            # Wrap in-text author/bib references in [bib_ref] tokens + summarize caption
            elif 'b' in annot_type:
                # Skip unidentified references
                if annot_type == 'b?':
                    text += ' '
                    start += len(subarray)
                    continue

                # Format identified references
                if annot_type in formatted_bibs:
                    bib_str = formatted_bibs[annot_type]
                else:
                    bib_str = format_bib(record, annot_type)
                    formatted_bibs[annot_type] = bib_str
                if bib_str:
                    text += ' [bib_ref] ' + bib_str + ' [/bib_ref] '

            # Keep figure/table content wrapped in [fig]/[table] tokens
            elif ('FIG_' in annot_type) or ('TAB_' in annot_type):
                at_figures = True
                fig_id = annot_type.split('_')[0].lower()+'_'+annot_type.split('_')[1]
                fig_name, caption = format_fig(record, fig_id, max_length=None)
                if fig_name and caption:
                    fig_str = fig_name + caption
                    # Check the figure hasn't already been added
                    added = any([re.sub(r'[:,()]', '', fig.strip()) in fig_name.lower() for fig in added_figures])
                    if 'continued' not in fig_str.lower() and not added:
                        added_figures += [fig_name.lower()]
                        tags = ['[fig]','[/fig]'] if 'FIG_' in annot_type else ['[table]','[/table]']
                        text += '\n' + tags[0] + ' ' + fig_str + ' ' + tags[1] + '\n'

            # Wrap formulae in [formula] tokens
            elif annot_type == 'FML':
                text += '\n[formula] ' + part + ' [/formula]\n'

            # Advance along the article
            start += len(subarray)

        except:
            # If there's any error in a part, just skip it
            start += len(subarray)
            continue

    # Further formatting
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\n# ', '\n\n# ', text)
    text = re.sub(r'\n## ', '\n\n## ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\[/fig_ref\] \.', '[/fig_ref].', text)
    text = re.sub(r'\[/bib_ref\] \.', '[/bib_ref].', text)
    return text

def process_s2orc(source_path, save_path, start=None, end=None):
    '''
    Pre-processing for full-text PubMed articles in S2ORC format.
    '''
    print(f'\nPre-processing PubMed articles in {source_path}.\n')
    if os.path.exists(save_path):
        print(f'File {save_path} already exists. Do you want to overwrite it? [y/n]')
        if input().lower() == 'y':
            os.remove(save_path)
    count = 0
    skipped = 0
    non_english = 0
    duplicates = 0
    total = 0
    with open(source_path, 'r') as f_in, open(save_path, 'a') as f_out:
        for line in tqdm(f_in):
            if start and count <= start:
                continue
            if end and count > end:
                break
            total += 1
            try: 
                # Filter out invalid entries
                record = json.loads(line)
                content = record.get('content')
                if not content:
                    skipped += 1
                    continue
                text = content.get('text')
                if not text:
                    skipped += 1
                    continue

                # Filter non-english articles
                language = detect_lang(text)
                if language != 'en':
                    non_english += 1
                    continue

                # Format article
                text = format_article(record)
                if not text:
                    skipped += 1
                    continue

                # Prepend title if given
                title = record.get('title')
                if title:
                    text = title + '\n\n' + text

                # Save article
                record.update({'text': text})
                record.pop('content')
                f_out.write(json.dumps(record) + '\n')
                count += 1
            except:
                skipped += 1
                continue
    print(f'Finished processing {count} out of {total} articles\
          \nRemoved {non_english} non-English articles.\
          \nRemoved {duplicates} duplicates.\
          \nSkipped {skipped} articles leading to errors. ')


def process_abstracts(source_path, save_path, start=None, end=None):
    ''' 
    Processing for PubMed abstracts.
    '''
    print(f'\nPre-processing text in {source_path}.\n')
    if os.path.exists(save_path):
        print(f'File {save_path} already exists. Do you want to overwrite it? [y/n]')
        if input().lower() == 'y':
            os.remove(save_path)
    total = 0
    count = 0
    non_english = 0
    duplicates = 0
    corpus_ids = set()
    with open(source_path, 'r') as f_in, open(save_path, 'a') as f_out:
        for line in tqdm(f_in):
            if start and total <= start:
                continue
            if end and total > end:
                break
            total += 1

            try: 
                record = json.loads(line)

                # Remove duplicates
                corpus_id = record.get('corpusid')
                if corpus_id and corpus_id in corpus_ids:
                    duplicates += 1
                    continue
                corpus_ids.add(corpus_id)

                text = record.get('text')
                if not text:
                    skipped += 1
                    continue

                # Filter non-english abstracts
                language = detect_lang(text)
                if language != 'en':
                    non_english += 1
                    continue
                
                # Prepend title if given
                title = record.get('title')
                if title:
                    text = title + '\n' + text

                # Cleaning up
                text = remove_urls(text)
                text = remove_references(text)

                record['text'] = text
                f_out.write(json.dumps(record) + '\n')
                count += 1

            except: 
                skipped += 1
                continue

    print(f'Finished processing {count} out of {total} articles\
          \nRemoved {non_english} non-English articles.\
          \nRemoved {duplicates} duplicates.\
          \nSkipped {skipped} articles leading to errors. ')

def split_s2orc(source_path):
    '''
    Split s2orc into PubMed and PubMedCentral subsets.
    '''
    pm_path = source_path.replace('.jsonl', '_pm.jsonl')
    pmc_path = source_path.replace('.jsonl', '_pmc.jsonl')

    with open(source_path, 'r') as f_in, open(pm_path, 'a') as f_pm, open(pmc_path, 'a') as f_pmc:
        for line in tqdm(f_in):
            record = json.loads(line)
            externalids = record.get('externalids')
            pm_id = externalids.get('PubMed')
            pmc_id = externalids.get('PubMedCentral')
            if pm_id and not pmc_id:
                f_pm.write(line)
            if pmc_id:
                f_pmc.write(line)


def train_test_split(source_path, split_ratio=0.03):
    '''
    Split a jsonl file into train and test sets.
    '''
    train_path = source_path.replace('.jsonl', '_train.jsonl')
    test_path = source_path.replace('.jsonl', '_test.jsonl')
    print(f'\nSplitting {source_path} into {train_path} and {test_path}.\n')
    train = 0
    test = 0
    with open(source_path, 'r') as f_in, open(train_path, 'a') as f_train, open(test_path, 'a') as f_test:
        for line in tqdm(f_in):
            if np.random.random() < split_ratio:
                f_test.write(line)
                test += 1
            else:
                f_train.write(line)
                train += 1
    print(f'Split {train} articles into {train_path} and {test} articles into {test_path}.')
            

def combine(source_paths, save_path):
    '''
    Combine s2orc and abstracts into a single file.
    '''
    paths = source_paths.split(',')
    print(f'\nCombining {len(paths)} files into {save_path}.\n')
    if os.path.exists(save_path):
        print(f'File {save_path} already exists. Do you want to overwrite it? [y/n]')
        if input().lower() == 'y':
            os.remove(save_path)
    for path in paths:
        subset = 'unknown'
        if 's2orc' in path:
            subset = 's2orc'
        elif 'abstracts' in path:
            subset = 'abstracts'
        elif 'guidelines' in path:
            subset = 'guidelines'
        print(f'Processing subset {subset} from {path}.')
        with open(path, 'r') as f_in, open(save_path, 'a') as f_out:
            for line in tqdm(f_in):
                record = json.loads(line)
                record['subset'] = subset
                f_out.write(json.dumps(record) + '\n')

def deduplicate(abstracts_path, s2orc_path):
    '''
    Remove all abstracts for which we already have a full-text version.
    '''
    # Get all corpus IDs in s2orc_path
    corpus_ids = set()
    with open(s2orc_path, 'r') as f_in:
        for line in f_in:
            record = json.loads(line)
            corpus_ids.add(record['corpusid'])

    # Remove all abstracts with corpus IDs in s2orc_path
    print(f'\nRemoving all abstracts with full-text versions in {s2orc_path} from {abstracts_path}.\n')
    removed = 0
    dedup_path = abstracts_path.replace('.jsonl', '_dedup.jsonl')
    with open(abstracts_path, 'r') as f_in, open(dedup_path, 'a') as f_out:
        for line in tqdm(f_in):
            record = json.loads(line)
            corpus_id = record.get('corpusid')
            if corpus_id and corpus_id in corpus_ids:
                removed += 1
                continue
            f_out.write(line)
    print(f'Removed {removed} abstracts with full-text versions in {s2orc_path} from {abstracts_path}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='s2orc',
        help="Dataset to process. Defaults to s2orc. Available: [s2orc, abstracts].")
    parser.add_argument(
        "--source_path", type=str,
        help="Path to jsonl file to process.")
    parser.add_argument(
        "--save_path", type=str,
        help="Path to save processed jsonl file.")
    parser.add_argument(
        "--start", type=int,
        default = None,
        help="Start index. Default: None")
    parser.add_argument(
        "--end", type=int,
        default = None,
        help="End index. Default: None")
    parser.add_argument(
        "--combine", 
        action='store_true',
        help="If passed as argument, combine files from source_path into save_path.")
    parser.add_argument(
        "--split", 
        action='store_true',
        help="If passed as argument, source_path is split into train and test sets.")
    parser.add_argument(
        "--split_s2orc",
        action='store_true',
        help="If passed as argument, split s2orc into PubMed and PubMedCentral subsets.")
    parser.add_argument(
        "--deduplicate",
        action='store_true',
        help="If passed as argument, remove all abstracts for which we already have a full-text version.")
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.03,
        help="Split ratio for train/test split. Default: 0.03"
    )
    args = parser.parse_args()

    if args.combine:
        combine(args.source_path, args.save_path)
        return
    
    elif args.split:
        train_test_split(args.source_path, args.split_ratio)
        return
    
    elif args.split_s2orc:
        split_s2orc(args.source_path)
        return
    
    elif args.deduplicate:
        deduplicate(args.source_path, args.save_path)
        return
    
    elif args.dataset == 's2orc':
        process_s2orc(args.source_path, args.save_path, args.start, args.end)

    elif args.dataset == 'abstracts': 
        process_abstracts(args.source_path, args.save_path, args.start, args.end)

    else:
        raise ValueError(f'Unknown dataset {args.dataset}. Available: [s2orc, abstracts].')

if __name__ == "__main__":
    main()