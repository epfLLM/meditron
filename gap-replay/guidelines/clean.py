'''
This script contains functions to pre-process 
clinical practice guidelines from different sources.

Guideline sources currently supported:
- AAFP (American Academy of Family Physicians): https://www.aafp.org
- CCO (Cancer Care Ontario): https://www.cancercareontario.ca/en/guidelines-advice
- CDC (Center for Disease Control and Prevention): https://www.cdc.gov/
- CMA (Canadian Medical Association): https://joulecma.ca/
- CPS (Canadian Paediatric Society): https://www.cps.ca
- drugs.com: https://www.drugs.com/
- GuidelineCentral: https://www.guidelinecentral.com/
- ICRC (International Committee of the Red Cross): http://icrc.org/
- IDSA (Infectious Diseases Society of America): https://www.idsociety.org/
- MAGIC (Making GRADE The Irresistible Choice): https://magicevidence.org/
- MayoClinic: https://www.mayoclinic.org/
- NICE (National Institute for Health and Care Excellence): https://www.nice.org.uk/guidance
- RCH (Royal Children's Hospital Melbourne): https://www.rch.org.au/clinicalguide/about_rch_cpgs/welcome_to_the_clinical_practice_guidelines/
- SPOR (Strategy for Patient-Oriented Research): https://sporevidencealliance.ca/key-activities/cpg-asset-map/cpg-database/ 
- WHO (World Health Organization): https://www.who.int/publications/who-guidelines
- WikiDoc: https://www.wikidoc.org/
'''

import json
import os
import re
import hashlib
import numpy as np
import argparse
from tqdm import tqdm
from langdetect import detect
import random


# -------------- Helper functions -------------- #


def read_jsonl(path):
    ''' 
    Read a jsonl file into a list of dictionaries.
    '''
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def detect_lang(text, sample_size=10000): 
    ''' 
    Detect language of a given text.
    '''
    sample = text if len(text) < sample_size else text[:sample_size]
    try:
        language = detect(sample)
    except:
        language = 'unknown'
    return language


def concatenate_sections(article): 
    '''
    Concatenate sections of an article into a single text.
    '''
    text = ''
    for section_name in article['content'].keys():
        section = article['content'][section_name]
        text += f'# {section_name}\n\n{section}'
    return text


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


def normalize_sections(text, max_hashes=3):
    '''
    Normalize section header hashes to min 1, max 3.
    '''
    if '\n#' not in text: 
        return text
    min_hashes = min([len(x)-1 for x in re.findall(r'\n#+', text)])
    text = re.sub(r'\n' + '#' * min_hashes, '\n#', text)
    text = re.sub(r'\n#{%d,}' % (max_hashes), '\n' + '#' * max_hashes, text)
    return text


def normalize_lists(text): 
    text = re.sub(r'\n\* ', '\n- ', text)
    text = re.sub(r'\n•', '\n-', text)
    text = re.sub(r'\no', '\n-', text)
    text = re.sub(r'\n', '\n-', text)
    text = re.sub(r'\n\+ ', '\n- ', text)
    text = re.sub(r'\n•', '\n-', text)
    text = text.replace('• ', '- ')
    text = re.sub(r'\* ', '- ', text)
    return text


def remove_weird_chars(text):
    text = re.sub(r'◆', '', text)
    text = re.sub(r'•', '', text)
    text = re.sub(r'', '', text)
    text = re.sub(r'▪', '', text)
    text = re.sub(r'■', '', text)
    text = re.sub(r'□', '', text)
    text = re.sub(r'\*-', '', text)
    text = re.sub(r'\n>', '\n', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'�', '', text)
    return text


def normalize_newlines(text):
    new_text = ''
    for line in text.split('\n'):
        line_an = re.sub(r'[^a-zA-Z ]', '', line).strip()
        if line_an == '':
            continue
        else: 
            new_text += line + '\n' 
    text = new_text
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\n#', '\n\n#', text)
    return text


def clean(text): 
    '''
    Common cleaning functions for all guidelines.
    - Remove URLs
    - Remove references []() and []
    - Normalize section hashes  
    - Normalize list formats
    - Remove weird characters
    - Normalize number of newlines
    '''
    text = remove_urls(text)
    text = remove_references(text)
    text = normalize_lists(text)
    text = remove_weird_chars(text)
    text = normalize_sections(text)
    text = normalize_newlines(text)
    return text.strip()


def truncate(text, starters=None, removers=None, stoppers=None):
    '''
    Truncate text so that:
    - it begins with the first line starting with a starter
    - it ends with the first line starting with a stopper
    - it removes all lines starting with a remover
    '''
    if starters:
        starters = [starter.lower() for starter in starters]
    if removers: 
        removers = [remover.lower() for remover in removers]
    if stoppers:
        stoppers = [stopper.lower() for stopper in stoppers]
    new_text = ''
    started = False
    for line in text.split('\n'):
        line_lower = line.lower().strip()
        line_clean = re.sub(r'#', '', line.lower()).strip()
        line_an = re.sub(r'[^a-zA-Z ]', '', line.lower()).strip()
        line_formats = [line, line_lower, line_clean, line_an]
        if starters and not started and any([lf.startswith(starter) for lf in line_formats for starter in starters]):
            new_text = line + '\n'
            started = True
        elif removers and any([lf.startswith(remover) for lf in line_formats for remover in removers]):
            continue
        elif stoppers and any([lf.startswith(stopper) for lf in line_formats for stopper in stoppers]):
            break
        elif line_an == '':
            continue
        elif stoppers and any([lf.startswith(stopper) for lf in line_formats for stopper in stoppers]):
            break
        else:
            new_text += line + '\n'
    return new_text.strip()


# -------------- Custom pre-processing functions -------------- #


# ================= AAFP =================

def process_aafp(guideline): 
    text = guideline['content'].strip()

    # Filter guidelines which support another institution's recommendations
    if 'The AAFP supports' in text:
        filter = [line for line in text.split('\n') if line.startswith('The AAFP supports')]
        if len(filter) > 0 and filter[0].strip().endswith('on this topic.'):
            return None
        
    title = guideline['title'].strip().replace(' | AAFP', '').replace(' - Choosing Wisely', '')
    starters = ['key recommendations', 
                'clinical preventive service recommendation', 'recommendation']
    removers = ['===', '---', '**[', '[', 'http']
    stoppers = [
        'references', 'related content', 'more about choosing wisely', 
        'more about practice guidelines', '*keyboard\_tab*', 'sources', 
        '*these recommendations are provided only', '*these guidelines are provided only']
    text = truncate(text, starters, removers, stoppers)
    new_text = title + '\n\n'
    for line in text.split('\n'):
        line_clean = re.sub(r'#', '', line.lower()).strip()
        if all([len(word) == 1 for word in line_clean.split()]):
            continue
        elif title.lower().startswith(line_clean): 
            continue
        elif '|' in line_clean or 'http' in line_clean:
            continue
        else: 
            new_text += line + '\n'
    text = clean(new_text)
    guideline = {'title': title, 'text': text, 'url': guideline['url']}
    return guideline


# ================= CCO =================


def process_cco(guideline): 
    text = guideline['text'].strip()
    starters = ['QUESTIONS', 'INTRODUCTION', 'INTENDED PURPOSE', "GUIDELINE OBJECTIVES"]
    removers = ["These guidelines recommendations have been endorsed",
                "This report is copyrighted by",
                "An assessment conducted in ",
                "Care has been taken in the preparation ",
                "This is a quick reference guide"]
    text = truncate(text, starters=starters, removers=removers)
    new_text = ''
    for line in text.split('\n'):
        line_an = re.sub(r'[^a-zA-Z]', '', line).strip()
        if line.strip().startswith('•'):
            new_text += '- ' + line[1:].strip() + '\n'
        elif len(line.strip().split()) < 3:
            continue
        elif line_an == '':
            continue
        else:
            new_text += re.sub(r'^o ', '- ', line) + '\n'
    text = clean(new_text)
    guideline = {'text': text}
    return guideline


# ================= CDC =================


def process_cdc_diseases(guideline):
    if guideline['content'].strip().split('\n')[0].strip() == '### Disease Directory':
        return None
    stoppers = ["More Information", "After Travel"]
    removers = ["insurance", " | CDC"]
    text = guideline['content'].strip()
    text = truncate(text, removers=removers, starters=['###'], stoppers=stoppers)

    title = text.split('\n')[0][3:].strip()
    new_text = guideline['title'] + '\n\n'
    for line in text.split('\n'):
        if len(line.strip().split()) < 3:
            continue
        elif any([s in line for s in removers]):
            continue
        else:
            n = re.sub('\+ ', '- ', line)
            n = re.sub('\*\*', '', n)
            n = re.sub('\*', '- ', n)
            new_text += n + '\n'
    guideline = {'title': title, 'text': new_text}
    return guideline


def process_cdc(guideline): 
    new_text = ''
    text = guideline['text'].strip()
    for line in text.split('\n'):
        line_clean = re.sub('[\•\#]', '', line.lower()).strip()
        line_an = re.sub(r'[^a-zA-Z]', '', line.lower()).strip()
        if line_clean == '': 
            continue
        elif line.startswith('q '):
            new_text += '- ' + line[2:].strip() + '\n'
        elif line_an.startswith('acknowledg'):
            break
        elif line.startswith('#') and all([c.isupper() for c in line_an]):
            new_text += ' '.join([w.capitalize() for w in line.split(' ')]) + '\n'
        else: 
            new_text += line.strip() + '\n'
    text = clean(new_text)
    text = re.sub(r'\n# -', '\n#', text)
    guideline = {'text': text, 'doi': guideline['doi']}
    return guideline


# ================= CMA =================


def process_cma(guideline): 
    text = guideline['content'].strip()
    title = text.split('\n')[0]
    starters = [
        'key information', '### key information', '### 1. what', 
        '### abstract', '### what', 'overview', 'introduction', 'preamble']
    removers = ['refer to', '===', '---', '* [', '[', 
                 '![', '|', 'table', 'figure', '+ [', 'footnote']
    stoppers = [
        '### selected references', 'selected references', '### references', '### authors’ statement', 
        'references', 'appendix', 'acknowledgments', 'acknowledgements', 'report a problem', 
        'list of abbreviations', 'additional tables', 'additional resources']
    text = truncate(text, starters, removers, stoppers)
    new_text = title + '\n\n'

    # Remove tables
    in_table = False
    for line in text.split('\n'):
        line_clean = line.strip().lower()
        if line_clean.startswith('table') or line_clean.startswith('figure'):
            in_table = True
        elif '|' in line:
            continue
        elif title in line:
            continue
        elif in_table: 
            if line == '': 
                in_table = False
            else: 
                continue
        else: 
            new_text += line + '\n'
    text = re.sub(r' ,', '', text)
    text = re.sub(r' \.', '.', text)
    text = clean(new_text)
    guideline = {'title': title, 'text': text}
    return guideline
    

def process_cma_pdfs(guideline): 
    text = guideline['text'].strip()
    stoppers = ['acknowledg', 'disclaimer', 'conflict of interest', 'funding']
    text = truncate(text, stoppers=stoppers)
    new_text = ''
    for line in text.split('\n'):
        line_clean = re.sub(r'[\#.]', '', line).strip().lower()
        if line_clean == '•' or line_clean == '':
            continue
        elif line_clean.isdigit():
            continue
        elif re.match(r'^\d+\.\s', line):
            line = '- ' + line[re.search(r'\d+\.\s', line).end():]
            new_text += line + '\n'
        else: 
            new_text += line + '\n'
    text = clean(new_text)
    guideline = {'text': text, 'doi': guideline['doi']}
    return guideline


# ================= CPS =================


def process_cps(guideline):
    stoppers = ["Acknowledgements", 'Selected resources', 'Current:  About CPS position']
    removers = ["|", "Figure", 'The Canadian Paediatric Society gives permission', 'Keywords:', 'Key words:']
    text = guideline['text'].strip()
    text = truncate(text, removers=removers, starters=['### Abstract'], stoppers=stoppers)

    new_text = ''
    for line in text.split('\n'):
        if len(line.strip().split()) < 3:
            continue
        elif any([s in line for s in removers]):
            continue
        else:
            new_text += line.strip() + '\n'
    text = clean(new_text)
    # if less than 3 lines, skip
    if len(text.split('\n')) < 3:
        return None
    guideline = {'text': text}
    return guideline


# ================= drugs.com =================


def process_drugs(guideline): 
    guideline['title'] = guideline['title'].split(' - ')[0].strip()
    text = guideline['content'].strip()
    removers = ['[Medical', '===', '---', '###', '* [', '[', 
                 'Always consult your healthcare provider', 
                 'Frequently asked', 'More about', 'Further information']
    text = '\n'.join([line for line in text.split('\n') if '|' not in line])
    text = truncate(text, removers=removers)
    text = re.sub(r'\*', '-', text)
    text = re.sub(r'\[|\]', '', text)
    text = re.sub(r'\([^\)]+\)', '', text).strip()
    text = re.sub(r'--', '', text)
    text = clean(text)
    guideline['text'] = text
    del guideline['content']
    return guideline


# ================= Guideline Central =================


def process_gc(guideline):
    title = guideline['title'].strip()
    text = guideline['text'].strip()
    if title == 'Title':
        title = ''
        text = text[6:]

    # Remove odd characters [" and "] from title and text
    title = re.sub(r'\["', '', title)
    title = re.sub(r'"\]', '', title)
    text = re.sub(r'\["', '', text)
    text = re.sub(r'"\]', '', text)
    starters = ['Document Objectives']
    removers = ['Publication Date', '(c)', 'www', 'http']
    stoppers = ['Disclaimer', 'Recommendation Grading', 'Source Citation']
    text = truncate(text, starters=starters, removers=removers, stoppers=stoppers)
    
    # Pre-process line by line
    new_text = ''
    for line in text.split('\n'):
        line = line.strip()
        line_an = re.sub(r'[^a-zA-Z0-9]', '', line).strip()
        if line == '(c)':
            continue
        elif line_an.isupper() and len(line_an) > 2:
            new_text += '\n' + '# ' + line + '\n'
        else: 
            new_text += line + '\n'
    text = clean(new_text)
    # Filter short or empty guidelines
    if len([line for line in text.split('\n') if len(line.split(' ')) > 10]) < 3:
        return None
    guideline = {'title' : title, 'text' : text}
    return guideline


# ================= ICRC =================


def process_icrc(guideline): 
    text = guideline['text']
    stoppers = ['acknowledgements', 'acknowledgments', 'contacts']
    text = truncate(text, stoppers=stoppers)
    new_text = ''
    for line in text.split('\n'):
        line_an = re.sub(r'[^a-zA-Z]', '', line).strip()
        if line_an == '':
            continue
        elif re.match(r'^\d+[a-zA-Z]+', line):
            new_text += '- ' + line[1:].strip() + '\n'
        else: 
            new_text += line + '\n'
    text = re.sub(r'\|', '', text)
    text = clean(new_text).strip()
    guideline = {'text': text}
    return guideline


# ================= IDSA =================


def process_idsa(guideline): 
    text = guideline['content'].strip()
    title = text.split('\n')[0]
    if 'This new guideline is currently in development' in text:
        return None
    starters = ['introduction', 'abstract']
    removers = ['---', '===', 'published', '[!', 'appropriate use criteria']
    stoppers = [
        'references', 'for more information', 'to view the full version', 'disclaimer', 
        'supplementary', 'acknowledgments', 'notes', 'to access a more user']
    text = truncate(text, starters, removers, stoppers)
    new_text = title + '\n\n'
    for line in text.split('\n'):
        line_clean = re.sub(r'#', '', line.lower()).strip()
        if line_clean.split('.')[0].isdigit():
            index_dot = line_clean.find('.')
            new_text += '- ' + line[index_dot+1:].strip() + '\n'
        elif title.lower().startswith(line_clean):
            continue
        else: 
            new_text += line + '\n'
    text = re.sub(r' ,', '', new_text)
    text = re.sub(r' \.', '.', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r' \)', '', text)
    text = re.sub(r'-\. ', '- ', text)
    text = clean(text)
    if len(text.split('\n')) < 5:
        return None
    guideline = {'title': title, 'text': text, 'url': guideline['url']}
    return guideline


# ================= MAGIC =================


def process_magic(guideline):
    # Text was loaded by chunks, remove some loading chunks
    text = ''
    chunks = guideline['content'].strip().split('Loading Data...\n')
    chunk_removers = ['Write remark here', 'Write header here']
    for _, chunk in enumerate(chunks): 
        if not any([x in chunk for x in chunk_removers]):
            text += chunk + '\n\n'

    # Format section headers
    new_text = ''
    sentences = text.split('\n')
    i = 0
    while i < len(sentences):
        if i == len(sentences)-3:
            new_text += '\n'.join(sentences[i:])
            break
        prev = sentences[i].strip()
        current = sentences[i+1].strip()
        next = sentences[i+2].strip()
        if prev.isdigit() and not current.isdigit() and next.isdigit():
            new_text += f'# {current}\n'
            i += 2
        else:
            new_text += current + '\n'
            i += 1

    starters = ['abstract', 'introduction']
    removers = [
        'updates', '===', '---', '![', 'please visit the', '< less', 'more >',
        'write general section text', 'loading data...', 'view section text', 'about this guideline']
    text = truncate(new_text, starters, removers)

    new_text = ''
    for line in text.split('\n'):
        line_clean = re.sub(r'#', '', line.lower()).strip()
        if line_clean == '' or '|' in line_clean:
            continue
        else: 
            new_text += line.strip() + '\n'

    # Character formatting
    text = re.sub(r'\n+', '\n', new_text)
    text = re.sub(r'\n\*\*', '\n## ', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'\n\(', '(', text)
    text = re.sub(r'\n\)', ')', text)
    text = re.sub(r'\n\;', ';', text)
    text = text.strip()

    # Remove sections starting with any of the following:
    removers = [
        'members', 'disclaimer', 'disclosure', 'funding', 'acknowledgements', 
        'acknowledgments', 'publisher', 'date of publication', 'authorship', 
        'publication approval', 'isbn', 'declarations of interest', 
        'external reviewers', 'contributors', 'previous versions in magicapp', 'references']
    new_text = ''
    for section in text.split('\n\n'):
        section_header = re.sub(r'[^a-zA-Z ]', '', section.split('\n')[0].lower()).strip()
        if any([section_header.startswith(x) for x in removers]):
            continue
        new_text += section + '\n\n'
    text = new_text

    # Remove any hashtags at the start of a line with > 7 words
    new_text = ''
    for line in text.split('\n'):
        line_clean = re.sub(r'[^a-zA-Z0-9 ]', '', line.lower()).strip()
        if line.startswith('!') or line_clean.strip().isdigit():
            continue
        elif line.startswith('#') and len(line.split(' ')) > 7:
            new_text += re.sub(r'#', '', line).strip() + '\n'
        else:
            new_text += line + '\n'

    text = clean(new_text)
    guideline = {'text': text}
    return guideline
    

# ================= MayoClinic =================


def process_mayo(guideline): 
    text = clean(concatenate_sections(guideline))
    text = '\n'.join([line for line in text.split('\n') if 'MayoClinic' not in line])
    guideline = {'title':guideline['name'], 'text':text}
    return guideline


# ================= NICE =================


def process_nice(guideline):
    content = {}
    excluders = ['advice', 'committee', 'implementation', 'team', 'update']
    for section_name, section in guideline['content'].items():
        if 'discussion' in section_name.lower():
            content[section_name] = section
        if not any(excluder in section_name.lower() for excluder in excluders):
            content[section_name] = section
    if len(content) == 0:
        return None
    guideline['content'] = content
    text = concatenate_sections(guideline)
    new_text = ''
    if guideline['name']: 
        new_text += guideline['name'] + '\n\n'
    if guideline['overview']:
        new_text += guideline['overview'] + '\n\n'
    new_text += clean(text)
    guideline = {'title': guideline['name'], 
                 'url': guideline['url'], 
                 'overview':guideline['overview'], 
                 'text': new_text}
    return guideline


# ================= RCH =================


def process_rch(guideline):
    guideline = {
        'title': guideline['name'],
        'url': guideline['url'],
        'text': guideline['content'],
    }
    return guideline


# ================= SPOR =================


def process_spor(guideline):
    text = guideline['text'].strip()
    stoppers = ['CONFLICT OF INTEREST']
    text = truncate(text, stoppers=stoppers)
    text = re.sub(r'\|', '', text)
    text = re.sub(r'---', '\n', text)
    new_text = ''
    for line in text.split('\n'):
        if len(line.strip().split()) < 3:
            continue
        else:
            new_text += line.strip() + '\n'
    text = clean(new_text)
    guideline = {'text': text}
    return guideline


# ================= WHO =================


def process_who(guideline):  # TO CHECK
    text = guideline['text'].strip()
    removers = ['|', 'Under the terms of this licence']
    stoppers = ['Acknowl']
    starters = ['Introduction ']
    text = truncate(text, starters=starters, removers=removers, stoppers=stoppers)
    new_text = ''
    for line in text.split('\n'):
        if len(line.strip().split()) < 3:
            continue
        else:
            n = re.sub(r'• ', '- ', line).strip()
            new_text += n + '\n'
    new_text = re.sub(r'', '', new_text)
    text = clean(new_text)
    guideline = {'text': text}
    return guideline


# ================= WikiDoc =================


def deduplicate_wikidoc(in_path, out_path): 
    with open(in_path, 'r') as f:
        raw_wikidoc = [json.loads(line) for line in f]
    print(f'Loaded {len(raw_wikidoc)} raw articles from wikidoc')
    # Deduplication
    wikidoc = {}
    num_same_text = 0
    num_duplicates = 0
    for article in raw_wikidoc:
        name = article['name']
        # If article already exists, append url
        if name in wikidoc:
            num_duplicates += 1
            if article['text'] == wikidoc[name]['text']:
                num_same_text += 1
            # Add url to list of urls
            wikidoc[name]['urls'].append(article['url'])
        # If article doesn't exist, add it
        else:
            wikidoc[name] = {
                'name':name, 
                'urls':[article['url']], 
                'text':article['text']
                }
    print(f'Found {num_duplicates} duplicates, {num_same_text} of which have the same text')
    print(f'Now have {len(wikidoc)} unique articles')

    # Remove duplicated sub-articles
    wikidoc = {k: v for k, v in sorted(wikidoc.items(), key=lambda item: len(item[0]), reverse=True)}
    num_removed = 0
    for name in list(wikidoc.keys()):
        if name.endswith('overview'):
            # Remove overview from name
            subject = re.sub('overview', '', name).strip()
            wikidoc[name]['name'] = subject
            # Find all sub-articles with name containing subject, and remove them
            for subname in list(wikidoc.keys()):
                if subname.startswith(subject) and not subname.endswith('overview'):
                    del wikidoc[subname]
                    num_removed += 1
    print(f'Removed {num_removed} sub-articles, {len(wikidoc)} articles remaining')
    with open(out_path, 'w') as f:
        for article in wikidoc.values():
            f.write(json.dumps(article) + '\n')


def process_wikidoc(guideline):
    text = guideline['text']
    # Remove all lines starting with: 
    removers = [
        'Editor', 'Associate Editor', 'Media:', 
        'Click here', 'For patient information click here', 
        'How to edit trial information', 'Template:', 'To go back to the main page']
    stoppers = ['external links']
    text = truncate(text, removers=removers, stoppers=stoppers)

    # Remove disclaimer until next section
    in_disclaimer = False
    new_text = ''
    for line in text.split('\n'):
        if line.startswith('Any recommendations found on these pages'):
            in_disclaimer = True
        elif line.startswith('#'):
            in_disclaimer = False
            new_text += line + '\n'
        elif line.startswith('CLASS'): 
            continue
        elif not in_disclaimer:
            new_text += line + '\n'
    text = new_text

    # Cut all text after 'Redirect to' (including)
    text = re.sub(r'Redirect to.*', '', text)
    text = clean(text)

    # If there are less than 5 lines with more than 10 words, skip
    if len([line for line in text.split('\n') if len(line.split(' ')) > 10]) < 3:
        return None
    url = None if len(guideline['urls']) == 0 else guideline['urls'][0]
    guideline = {
        'title': guideline['name'],
        'url': url,
        'text': text
    }
    return guideline


# -------------- General utility functions -------------- #

global PROCESSORS

PROCESSORS = {
    'aafp' : process_aafp,
    'cco' : process_cco,
    'cdc' : process_cdc,
    'cdc_diseases' : process_cdc_diseases,
    'cma' : process_cma,
    'cma_pdfs' : process_cma_pdfs,
    'cps' : process_cps,
    'drugs' : process_drugs,
    'gc' : process_gc,
    'icrc' : process_icrc,
    'idsa' : process_idsa,
    'magic' : process_magic,
    'mayo' : process_mayo,
    'nice' : process_nice,
    'rch' : process_rch,
    'spor' : process_spor,
    'who' : process_who,
    'wikidoc' : process_wikidoc
}


def _hash_for_dedup(text, dedup_chars=500):
    '''
    Hashes the text to be used for deduplication.

    :param text: text to be hashed
    :param dedup_chars: number of characters to use for deduplication (default: 100)
    '''
    sample = text if len(text) < dedup_chars else text[:dedup_chars]
    dedup_str = re.sub(r'[^a-zA-Z0-9]', '', sample.lower())
    return dedup_str


def process_guidelines(source, in_path, out_path, english_only=True):
    '''
    Apply a processing function to all guidelines from a source. 
    
    :param source: name of the guideline source
    :param in_path: path to the jsonl file containing the guidelines
    :param out_path: path to the output jsonl file
    :param english_only: if True, only keep English-language guidelines (default: True)
    '''
    guidelines = read_jsonl(in_path)
    _process = PROCESSORS[source]
    non_english = 0
    filtered = 0
    duplicates = 0
    dedup_strings = set()
    processed_guidelines = []
    for g in tqdm(guidelines, f'Processing {source} guidelines'):
        new_guid = _process(g)

        if not new_guid:
            filtered += 1
            continue

        if english_only: 
            if detect_lang(new_guid['text']) != 'en':
                non_english += 1
                continue

        dedup_str = _hash_for_dedup(new_guid['text'])
        if dedup_str in dedup_strings:
            duplicates += 1
            continue
        dedup_strings.add(dedup_str)
        guid = {'source': source.split('_')[0], 
                'title': new_guid.get('title', None),
                'clean_text': new_guid['text'],
                'raw_text': g.get('text', g.get('content', None)),
                'url': new_guid.get('url', None),
                'overview': new_guid.get('overview', None),
                }
        processed_guidelines.append(guid)
    with open(out_path, 'w') as f_out:
        f_out.write('\n'.join([json.dumps(guid) for guid in processed_guidelines]))
    if non_english > 0:
        print(f'Skipped {non_english} non-english guidelines.')
    if filtered > 0: 
        print(f'Filtered out {filtered} guidelines during processing.')
    if duplicates > 0:
        print(f'Filtered out {duplicates} duplicates.')
    print(f'Processed {len(processed_guidelines)} guidelines from {source}.')


def print_statistics(in_path): 
    '''
    Divide articles by source, for each source print: 
    - number of articles
    - Average number of lines
    - Average number of words
    - Total number of words
    '''
    articles = read_jsonl(in_path)
    sources = {}
    for article in articles:
        source = article['source']
        if source not in sources:
            sources[source] = []
        sources[source].append(article)
    print('='*50)
    total_articles = len(articles)
    total_lines = 0
    total_words = 0
    sources = {k: v for k, v in sorted(sources.items(), key=lambda item: item[0])}
    for source in sources:
        print(f'\nSource: {source}')
        articles = sources[source]
        num_lines = [len(article['clean_text'].split('\n')) for article in articles]
        num_words = [len(article['clean_text'].split(' ')) for article in articles]
        total_lines += np.sum(num_lines)
        total_words += np.sum(num_words)
        print(f'Number of guidelines: {len(articles)}')
        print(f'Average number of lines: {np.mean(num_lines):.2f}')
        print(f'Average number of words: {np.mean(num_words):.2f}')
        print(f'Total number of words: {np.sum(num_words):,}')
        print(f'Total number of lines: {np.sum(num_lines):,}')
        print('\n'+'='*50)
    print(f'\nTotal number of guidelines: {total_articles}')
    print(f'Total number of lines: {total_lines:,}')
    print(f'Total number of words: {total_words:,}')


def combine_guidelines(dir_path, out_path, sources=None, min_chars=10): 
    '''
    Combine all guidelines from a directory into a single file.
    '''
    guidelines = []
    k = "clean_text"
    jsonl_files = sorted([file for file in os.listdir(dir_path) if (file.endswith('.jsonl') and 'guideline' not in file)])
    for file in jsonl_files:
        if sources and not any([s in file for s in sources]):
            continue
        source_guidelines = read_jsonl(os.path.join(dir_path, file))
        source_guidelines = [g for g in source_guidelines if g[k] and len(g[k]) > min_chars]
        guidelines.extend(source_guidelines)
    with open(out_path, 'w') as f_out:
        f_out.write('\n'.join([json.dumps(guideline) for guideline in guidelines]))


def add_guidelines(previous_path, add_path, new_path, overwrite=True):
    ''' Combine guidelines files. '''
    prev_guidelines = []
    if previous_path: 
        prev_guidelines = read_jsonl(previous_path)
    add_guidelines = read_jsonl(add_path)
    if overwrite: # Remove all previous guidelines of that source
        source = add_guidelines[0]['source']
        prev_guidelines = [guideline for guideline in prev_guidelines if guideline['source'] != source]
    with open(new_path, 'w') as f_out:
        new_guidelines = prev_guidelines + add_guidelines
        for guideline in new_guidelines:
            f_out.write(json.dumps(guideline) + '\n')


def create_samples(in_path, out_dir, num_samples=100):
    ''' 
    Sample randomized articles for each guideline source for inspection.
    '''
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    with open(in_path, 'r') as f_in:
        guidelines = [json.loads(line) for line in f_in]
    sources = []
    for guideline in guidelines:
        if guideline['source'] not in sources:
            sources.append(guideline['source'])
    for source in sources:
        idx_source = [i for i, guideline in enumerate(guidelines) if guideline['source'] == source]
        size = min(num_samples, len(idx_source))
        random_idx = np.random.choice(idx_source, size=size, replace=False)
        random_samples = [guidelines[i]['text'] for i in random_idx]
        text = ''
        for i, sample in enumerate(random_samples):
            text += '\n\n\n\n' + '=' * 20 + f' Sample {i+1} ' + '=' * 20 + '\n\n\n\n' + sample
        out_path = os.path.join(out_dir, source+'.txt')
        with open(out_path, 'w') as f_out:
            f_out.write(text)


def add_guideline_ids(path): 
    '''
    Add a unique ID to each guideline in the dataset. 
    '''
    with open(path, 'r') as f:
        guidelines = [json.loads(line) for line in f.readlines()]
    os.remove(path)
    sources = sorted(set([g['source'] for g in guidelines]))
    for source in sources:
        new_guidelines = []
        source_guidelines = [g for g in guidelines if g['source'] == source]
        for i, g in enumerate(source_guidelines):
            id = hashlib.sha1((source + str(i)).encode()).hexdigest()
            new_g = {'id': id}
            new_g.update(g)
            new_guidelines.append(new_g)
        with open(path, 'a') as f:
            for g in new_guidelines:
                f.write(json.dumps(g) + '\n')


def split_guidelines(in_path):
    '''
    Split guidelines into train/validation/test sets. 
        Train: 90% of all sources except Uptodate
        Validation: 5% of all sources except Uptodate
        Test: 5% of all sources except Uptodate
    ''' 
    # Split guidelines into train, val and test; 
    guidelines = read_jsonl(in_path)
    train = random.sample(guidelines, int(len(guidelines)*0.9))
    non_train = [g for g in guidelines if g not in train]
    val = random.sample(non_train, int(len(non_train)*0.5))
    test = [g for g in non_train if g not in val]

    # Save guidelines to files;
    train_path = in_path.replace('.jsonl', '_train.jsonl')
    val_path = in_path.replace('.jsonl', '_val.jsonl')
    test_path = in_path.replace('.jsonl', '_test.jsonl')
    sources = sorted(set([g['source'] for g in guidelines]))
    for path, data in zip([train_path, val_path, test_path], [train, val, test]):
        with open(path, 'w') as f:
            for source in sources: 
                source_guidelines = [g for g in data if g['source'] == source]
                for g in source_guidelines:
                    f.write(json.dumps(g) + '\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        help="Path to directory with raw .jsonl guidelines.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to directory with processed .jsonl guidelines.")
    parser.add_argument(
        "--process",
        action="store_true",
        help="If passed as argument, will process all guidelines in raw_dir. \
            If not given, will combine all processed guidelines in save_dir, add IDs and split into train/val/test.")
    
    args = parser.parse_args()
    if args.process:
        if not os.path.exists(args.raw_dir): 
            raise ValueError(f'{args.raw_dir} does not exist')
        print(f'Processing guidelines from {len(PROCESSORS.keys())} sources in {args.raw_dir}')
        for i, source in enumerate(PROCESSORS.keys()):
            in_path = f'{args.raw_dir}/{source}.jsonl'
            out_path = f'{args.save_dir}/{source}.jsonl'
            if not os.path.exists(in_path):
                print(f'[{i} | {len(PROCESSORS.keys())}] {source} guidelines not found at {in_path}')
                continue
            if os.path.exists(out_path):
                print(f'[{i} | {len(PROCESSORS.keys())}] {source} guidelines already processed, skipping')
                continue
            print(f'[{i} | {len(PROCESSORS.keys())}] Processing {source} guidelines')
            process_guidelines(source, in_path, out_path)
    else:
        guid_path = args.save_dir + 'guidelines.jsonl'
        combine_guidelines(args.save_dir, guid_path)
        print_statistics(guid_path)
        add_guideline_ids(guid_path)
        split_guidelines(guid_path) 
        