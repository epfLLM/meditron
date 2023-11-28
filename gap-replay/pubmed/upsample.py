import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import argparse
import json
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from pubmed.approved_journals import APPROVED_JOURNALS


FILTER_CONSTANT = -1000000
PUBLICATION_TYPE_FACTORS = {
    "Guideline": {
        "filter": 1,
        "upsample": 1
    },
    "Practice Guideline": {
        "filter": 1,
        "upsample": 1
    },
    "Patient Education Handout": {
        "filter": 1,
        "upsample": 1
    },
    "Meta-Analysis": {
        "filter": 1,
        "upsample": 1
    },
    "Systematic Review": {
        "filter": 1,
        "upsample": 0.8
    },
    "Clinical Trial, Phase IV": {
        "filter": 1,
        "upsample": 0.8
    },
    "Clinical Trial, Phase III": {
        "filter": 1,
        "upsample": 0.6
    },
    "Clinical Trial, Phase II": {
        "filter": 1,
        "upsample": 0.4
    },
    "Clinical Trial, Phase I": {
        "filter": 1,
        "upsample": 0.2
    },
    "Randomized Controlled Trial": {
        "filter": 1,
        "upsample": 0.5
    },
    "Review": {
        "filter": 1,
        "upsample": 0.5
    },
    "Observational Study": {
        "filter": 1,
        "upsample": 0.5
    },
    "Comparative Study": {
        "filter": 1,
        "upsample": 0.5
    },
    "Clinical Study": {
        "filter": 1,
        "upsample": 0.4
    },
    "Observational Study, Veterinary":
    {
        "filter": 0,
        "upsample": FILTER_CONSTANT
    },
    "Case Reports": {
        "filter": 0,
        "upsample": 0
    },
    "Editorial": {
        "filter": 0,
        "upsample": 0.1
    },
    "Letter": {
        "filter": 0,
        "upsample": 0.1
    },
    "Comment": {
        "filter": 0,
        "upsample": 0.1
    },
    "Retracted Publication": {
        "filter": 0,
        "upsample": FILTER_CONSTANT
    },
    "Retraction of Publication": {
        "filter": 0,
        "upsample": FILTER_CONSTANT
    },
    "Preprint": {
        "filter": 0,
        "upsample": FILTER_CONSTANT
    },
}

MESH_FACTORS = {
    "Animals": {
        "filter": 0,
        "upsample": FILTER_CONSTANT
    },
}
MESH_FACTORS = defaultdict(lambda: {"filter": 1, "upsample": 0}, MESH_FACTORS)


CURRENT_DATE = datetime.strptime("2023-07-15", "%Y-%m-%d")  # date of scraping
FIRST_CUTOFF_DATE = CURRENT_DATE - timedelta(days=365*5.5)
SECOND_CUTOFF_DATE = CURRENT_DATE - timedelta(days=365*10)
DATE_FACTORS = {
    "new": {
        "filter": 1,
        "upsample": 1
    },
    "middle": {
        "filter": 1,
        "upsample": 0.2
    },
    "old": {
        "filter": 0,
        "upsample": 0
    }
}

CITATION_FACTORS = {
    "top": { # top 25% of citations
        "filter": 1,
        "upsample": 1
    },
    "middle": { # middle 50% of citations
        "filter": 1,
        "upsample": 0.5
    },
    "bottom": { # bottom 25% of citations
        "filter": 0,
        "upsample": 0
    }
}

JOURNAL_FACTORS = {
    "approved": {
        "filter": 1,
        "upsample": 1
    },
    "non-approved": {
        "filter": 0,
        "upsample": 0
    }
}


def include_decision(article, citation_thresholds):
    """
    DEPRECATED: this was used for the old "Quality FILTER" scheme, which is now partially included in
    the "Quality UPSAMPLING" options.
    """
    # Whether the article has at least one publication type that is covered in our list
    publication_covered = 0
    for publicationtype in article["publicationtype"]:
        if publicationtype in PUBLICATION_TYPE_FACTORS:
            publication_covered = 1
            if PUBLICATION_TYPE_FACTORS[publicationtype]["filter"] == 0:
                include_decision.filtered_from_publication_type += 1
                return False
    if publication_covered == 0:
        include_decision.filtered_from_publication_notcovered += 1
        return False

    for mesh in article["mesh"]:
        if MESH_FACTORS[mesh]["filter"] == 0:
            include_decision.filtered_from_mesh += 1
            return False

    try: 
        if datetime.strptime(article["publicationDate"], "%Y-%m-%d") < SECOND_CUTOFF_DATE:
            if DATE_FACTORS["old"]["filter"] == 0:
                include_decision.filtered_from_date += 1
                return False
        elif datetime.strptime(article["publicationDate"], "%Y-%m-%d") < FIRST_CUTOFF_DATE:
            if DATE_FACTORS["middle"]["filter"] == 0:
                include_decision.filtered_from_date += 1
                return False
        else:
            if DATE_FACTORS["new"]["filter"] == 0:
                include_decision.filtered_from_date += 1
                return False
        publication_age = (CURRENT_DATE - datetime.strptime(article["publicationDate"], "%Y-%m-%d")).days // 365 + 1
        n_citations_normalized = article["citationCount"] / publication_age
        if n_citations_normalized <= citation_thresholds["bottom25"]:
            if CITATION_FACTORS["bottom"]["filter"] == 0:
                include_decision.filtered_from_citation += 1
                return False
        elif n_citations_normalized <= citation_thresholds["top25"]:
            if CITATION_FACTORS["middle"]["filter"] == 0:
                include_decision.filtered_from_citation += 1
                return False
        else:
            if CITATION_FACTORS["top"]["filter"] == 0:
                include_decision.filtered_from_citation += 1
                return False
    except TypeError:  # publicationDate is None (i.e. missing metadata)
        include_decision.filtered_from_nometadata += 1
        return False

    translation = {ord(ch): '' for ch in ":,.-"}
    if article["venue"].lower().translate(translation) in APPROVED_JOURNALS:
        if JOURNAL_FACTORS["approved"]["filter"] == 0:
            include_decision.filtered_from_journal += 1
            return False
    else:
        if JOURNAL_FACTORS["non-approved"]["filter"] == 0:
            include_decision.filtered_from_journal += 1
            return False    

    return True

include_decision.filtered_from_nometadata = 0
include_decision.filtered_from_publication_type = 0
include_decision.filtered_from_publication_notcovered = 0
include_decision.filtered_from_mesh = 0
include_decision.filtered_from_date = 0
include_decision.filtered_from_citation = 0
include_decision.filtered_from_journal = 0


def compute_factor(article, citation_thresholds):
    """
    Tweak this function and the dictionaries at the top of the file
    to change the UPSAMPLING OPTION.    
    """
    factor = 0
    no_publication_type = True

    for publicationtype in article["publicationtype"]:
        if publicationtype in PUBLICATION_TYPE_FACTORS:
            no_publication_type = False
            factor += PUBLICATION_TYPE_FACTORS[publicationtype]["upsample"]

    for mesh in article["mesh"]:
        factor += MESH_FACTORS[mesh]["upsample"]

    translation = {ord(ch): '' for ch in ":,.-"}
    if article["venue"].lower().translate(translation) in APPROVED_JOURNALS:
        factor += JOURNAL_FACTORS["approved"]["upsample"]
    else:
        factor += JOURNAL_FACTORS["non-approved"]["upsample"]

    # if article["uptodate_reference"]:
    #     factor += 1

    # if article["cochrane_reference"]:
    #     factor += 1

    try:
        # Only add date factor for SELECTED articles
        if factor > 0:
            if datetime.strptime(article["publicationDate"], "%Y-%m-%d") < SECOND_CUTOFF_DATE:
                # if no_publication_type:
                    # return FILTER_CONSTANT
                factor += DATE_FACTORS["old"]["upsample"]
            elif datetime.strptime(article["publicationDate"], "%Y-%m-%d") < FIRST_CUTOFF_DATE:
                # if no_publication_type:
                    # return FILTER_CONSTANT
                factor += DATE_FACTORS["middle"]["upsample"]
            else:
                factor += DATE_FACTORS["new"]["upsample"]

            publication_age = (CURRENT_DATE - datetime.strptime(article["publicationDate"], "%Y-%m-%d")).days // 365 + 1
            n_citations_normalized = article["citationCount"] / publication_age
            if n_citations_normalized <= citation_thresholds["bottom25"]:
                # if no_publication_type:
                    # return FILTER_CONSTANT
                factor += CITATION_FACTORS["bottom"]["upsample"]
            elif n_citations_normalized <= citation_thresholds["top25"]:
                # if no_publication_type:
                    # return FILTER_CONSTANT
                factor += CITATION_FACTORS["middle"]["upsample"]
            else:
                factor += CITATION_FACTORS["top"]["upsample"]  
    except TypeError:   # if the publicationDate is None (i.e. missing metadata)
        return FILTER_CONSTANT

    return factor


def filter(source_path, output_dir, citation_thresholds):
    """
    Filter the articles according to the quality thresholds.

    DEPRECATED: this corresponded to the old "Quality FILTER" scheme, which is now partially included in
    the "Quality UPSAMPLING" options.
    """
    with open(source_path, 'r') as f_in, open(output_dir + "filtered.jsonl", 'w') as f_out:
        errors = 0
        for i, line in tqdm(enumerate(f_in), total=4700000, desc="Filtering..."):
            record = json.loads(line)
            try:
                if include_decision(record, citation_thresholds):
                    f_out.write(line)
            except KeyError:
                errors += 1
                continue
                
    print(f"Errors: {errors}")

    total_articles  = 899631
    remaining_articles = total_articles
    print(f"Number of articles before filtering: {remaining_articles}")
    print(F"Number of articles with metadata: {remaining_articles - errors - include_decision.filtered_from_nometadata} (removed {errors+include_decision.filtered_from_nometadata}, {(errors+include_decision.filtered_from_nometadata) / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= (errors + include_decision.filtered_from_nometadata)
    print(f"Number of articles after filtering from publication type (removing Case Reports, Editorials, Comments, Letters, Redacted and preprints): {remaining_articles-include_decision.filtered_from_publication_type} (filtered {include_decision.filtered_from_publication_type}, {include_decision.filtered_from_publication_type / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_publication_type
    print(f"Number of articles after filtering from publication not covered: {remaining_articles-include_decision.filtered_from_publication_notcovered} (filtered {include_decision.filtered_from_publication_notcovered}, {include_decision.filtered_from_publication_notcovered / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_publication_notcovered
    print(f"Number of articles after filtering from mesh (removing Animals): {remaining_articles-include_decision.filtered_from_mesh} (filtered {include_decision.filtered_from_mesh}, {include_decision.filtered_from_mesh / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_mesh
    print(f"Number of articles after filtering from date: {remaining_articles-include_decision.filtered_from_date} (filtered {include_decision.filtered_from_date}, {include_decision.filtered_from_date / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_date
    print(f"Number of articles after filtering from citation: {remaining_articles-include_decision.filtered_from_citation} (filtered {include_decision.filtered_from_citation}, {include_decision.filtered_from_citation / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_citation
    print(f"Number of articles after filtering from journal: {remaining_articles-include_decision.filtered_from_journal} (filtered {include_decision.filtered_from_journal}, {include_decision.filtered_from_journal / remaining_articles * 100:.2f}% of remaining)")
    remaining_articles -= include_decision.filtered_from_journal

    print(f"Final number of articles after filtering: {remaining_articles} ({remaining_articles / total_articles * 100:.2f}% of original)")
            


def upsample(source_path, output_dir, citation_thresholds, upfold, test_size=0.03):
    """
    Upsample the articles according to the quality thresholds. First computes all the factors, then converts them to probabilities
    and sample with replacement according to these probabilities.
    """
    np.random.seed(42)

    ## Step 1: Compute factors for each article, filter out the articles with factor 0 (i.e. the ones explicitly filtered out, such as animal studies)
    ## and split the remaining ones into a train and test set
    train_path = output_dir + "train.jsonl"
    test_path = output_dir + "test.jsonl"
    with open(source_path, 'r') as f_in, open(train_path, 'w') as f_train, open(test_path, 'w') as f_test:
        factors = []
        errors = 0
        n_train = 0
        n_test = 0
        n_filtered = 0
        for line in tqdm(f_in, total=4700000, desc="Computing UpSampling factors..."):
            record = json.loads(line)
            try:
                factor = compute_factor(record, citation_thresholds)
            except (KeyError, TypeError):
                errors += 1
                factor = FILTER_CONSTANT

            if factor >= 0:
                if np.random.rand() < test_size:
                    f_test.write(line)
                    n_test += 1
                else:
                    f_train.write(line)
                    n_train += 1
                    factors.append(factor)
            else:
                n_filtered += 1

    n_total = n_train + n_test + n_filtered
    with open(output_dir + "log.txt", 'a') as f_out:
        f_out.write(f"Errors: {errors}\n")
        f_out.write(f"Number of articles (with metadata) before filtering: {n_total}\n")
        f_out.write(f"Number of articles after filtering: {n_total - n_filtered} ({(n_total - n_filtered) / n_total * 100:.2f}% of all)\n")
        remaining = n_total - n_filtered
        f_out.write(f"Number of articles in train set: {n_train} ({n_train / remaining * 100:.2f}% of after filtering)\n")
        f_out.write(f"Number of articles in test set: {n_test} ({n_test / remaining * 100:.2f}% of after filtering)\n\n")

    factors = np.array(factors)

    ## Step 2: Convert factors to counts, and save the counts in a file
    counts = method1(factors, upfold)       # change here the UPSAMPLING METHOD
    save_counts(counts, train_path, output_dir)


def method1(factors, upfold):
    counts = 1 + factors * upfold

    for i in tqdm(range(len(counts)), desc="Converting factors to counts"):
        if np.random.rand() < counts[i] - int(counts[i]):
            counts[i] = int(counts[i]) + 1
        else:
            counts[i] = int(counts[i])

    return counts


def method2(factors, upfold):
    counts = factors * upfold

    for i in tqdm(range(len(counts)), desc="Converting factors to counts"):
        if np.random.rand() < counts[i] - int(counts[i]):
            counts[i] = int(counts[i]) + 1
        else:
            counts[i] = int(counts[i])

    return counts


def method3(factors, upfold, clamp=5):
    counts = np.exp(factors) * upfold
    for i in tqdm(range(len(counts)), desc="Converting factors to counts"):
        if np.random.rand() < counts[i] - int(counts[i]):
            counts[i] = min(int(counts[i]) + 1, clamp)
        else:
            counts[i] = min(int(counts[i]), clamp)

    return counts

def save_counts(counts, train_path, output_dir):
    # print in log file
    log_path = output_dir + "log.txt"
    with open(log_path, 'a') as f_out:
        f_out.write("Statistics about counts:\n")
        f_out.write(f"Mean: {np.mean(counts)}\n")
        f_out.write(f"Std: {np.std(counts)}\n")
        f_out.write(f"Min: {np.min(counts)}\n")
        f_out.write(f"Max: {np.max(counts)}\n")
        f_out.write(f"10th percentile: {np.quantile(counts, 0.10)}\n")
        f_out.write(f"25th percentile: {np.quantile(counts, 0.25)}\n")
        f_out.write(f"Median: {np.median(counts)}\n")
        f_out.write(f"75th percentile: {np.quantile(counts, 0.75)}\n")
        f_out.write(f"90th percentile: {np.quantile(counts, 0.9)}\n")
        f_out.write(f"95th percentile: {np.quantile(counts, 0.95)}\n")
        f_out.write(f"99th percentile: {np.quantile(counts, 0.99)}\n")
        f_out.write("\n\n")
    
    # create plots of distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(counts)
    plt.xlabel("Counts")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir + "count_distribution.png")

    plt.figure(figsize=(10, 5))
    sns.histplot(counts, log_scale=(False, True))
    plt.xlabel("Counts")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir + "count_distribution_log.png")

    with open(train_path, 'r') as f_in, open(output_dir + "train_upsampled.jsonl", 'w') as f_out:
        for i, line in tqdm(enumerate(f_in), total=len(counts), desc="Writing output..."):
            for _ in range(int(counts[i])):
                f_out.write(line)


def compute_statistics(file_dir, log_dir, citation_thresholds):
    articles_per_publicationtype = defaultdict(int)
    articles_per_datecategory = defaultdict(int)
    articles_per_citationscategory = defaultdict(int)
    articles_approvedjournals = 0
    # articles_uptodate_refs = 0
    # articles_cochrane_refs = 0
    narticles = 0


    translation = {ord(ch): '' for ch in ":,.-"}
    with open(file_dir, 'r') as f_in:
        for i, line in tqdm(enumerate(f_in), total=4700000, desc="Computing statistics..."):
            article = json.loads(line)
            try:
                covered = 0
                for publicationtype in article["publicationtype"]:
                    if publicationtype in PUBLICATION_TYPE_FACTORS:
                        covered = 1
                        articles_per_publicationtype[publicationtype] += 1
                if covered == 0:
                    articles_per_publicationtype["other"] += 1

                narticles += 1
                if datetime.strptime(article["publicationDate"], "%Y-%m-%d") < SECOND_CUTOFF_DATE:
                    articles_per_datecategory["old"] += 1
                elif datetime.strptime(article["publicationDate"], "%Y-%m-%d") < FIRST_CUTOFF_DATE:
                    articles_per_datecategory["middle"] += 1
                else:
                    articles_per_datecategory["new"] += 1
                

                publication_age = (CURRENT_DATE - datetime.strptime(article["publicationDate"], "%Y-%m-%d")).days // 365 + 1
                n_citations_normalized = article["citationCount"] / publication_age

                if n_citations_normalized <= citation_thresholds["bottom25"]:
                    articles_per_citationscategory["bottom25"] += 1
                elif n_citations_normalized <= citation_thresholds["top25"]:
                    articles_per_citationscategory["middle50"] += 1
                else:
                    articles_per_citationscategory["top25"] += 1

                if article["venue"].lower().translate(translation) in APPROVED_JOURNALS:
                    articles_approvedjournals += 1

                # if article["uptodate_reference"]:
                #     articles_uptodate_refs += 1

                # if article["cochrane_reference"]:
                #     articles_cochrane_refs += 1

            except:
                continue
                

    with open(log_dir, 'a') as f_out:
        f_out.write("Statistics about articles:\n")
        f_out.write(f"Number of articles: {narticles}\n")
        for k, v in sorted(articles_per_publicationtype.items(), key=lambda item: item[0]):
            f_out.write(f"Number of articles with publication type {k}: {v} ({v/narticles*100:.2f}%)\n")
        f_out.write("----------------------------\n")
        f_out.write(f"Number of articles with date < 5.5 years {articles_per_datecategory['new']} ({articles_per_datecategory['new']/narticles*100:.2f}%)\n")
        f_out.write(f"Number of articles with date between 5.5 and 10 years {articles_per_datecategory['middle']} ({articles_per_datecategory['middle']/narticles*100:.2f}%)\n")
        f_out.write(f"Number of articles with date > 10 years {articles_per_datecategory['old']} ({articles_per_datecategory['old']/narticles*100:.2f}%)\n")
        f_out.write("----------------------------\n")

        f_out.write(f"Number of articles with citation count in top 25%: {articles_per_citationscategory['top25']} ({articles_per_citationscategory['top25']/narticles*100:.2f}%)\n")
        f_out.write(f"Number of articles with citation count in middle 50%: {articles_per_citationscategory['middle50']} ({articles_per_citationscategory['middle50']/narticles*100:.2f}%)\n")
        f_out.write(f"Number of articles with citation count in bottom 25%: {articles_per_citationscategory['bottom25']} ({articles_per_citationscategory['bottom25']/narticles*100:.2f}%)\n")
        f_out.write("----------------------------\n")

        f_out.write(f"Number of articles in approved journals: {articles_approvedjournals} ({articles_approvedjournals/narticles*100:.2f}%)\n")
        f_out.write(f"Number of articles not in approved journals: {narticles - articles_approvedjournals} ({(narticles - articles_approvedjournals)/narticles*100:.2f}%)\n")
        f_out.write("----------------------------\n")

        # f_out.write(f"Number of articles with UpToDate references: {articles_uptodate_refs} ({articles_uptodate_refs/narticles*100:.2f}%)\n")
        # f_out.write(f"Number of articles without UpToDate references: {narticles - articles_uptodate_refs} ({(narticles - articles_uptodate_refs)/narticles*100:.2f}%)\n")
        # f_out.write("----------------------------\n")

        # f_out.write(f"Number of articles with Cochrane references: {articles_cochrane_refs} ({articles_cochrane_refs/narticles*100:.2f}%)\n")
        # f_out.write(f"Number of articles without Cochrane references: {narticles - articles_cochrane_refs} ({(narticles - articles_cochrane_refs)/narticles*100:.2f}%)\n")



def compute_citation_thresholds(source_path, output_path):
    """
    Find the number of citations corresponding to the top 25%, middle 50%, and bottom 25% of citations.
    """
    with open(source_path, 'r') as f_in:
        citation_counts = []
        for i, line in tqdm(enumerate(f_in), total=4700000, desc="Counting citation quantiles..."):
            record = json.loads(line)
            try:
                publication_age = (CURRENT_DATE - datetime.strptime(record["publicationDate"], "%Y-%m-%d")).days // 365 + 1
                n_citations_normalized = record["citationCount"] / publication_age
            except: # if the publicationDate is None or missing
                continue
            citation_counts += [n_citations_normalized]
    
    citation_counts = np.array(citation_counts)
    top_threshold = np.quantile(citation_counts, 0.75)
    middle_threshold = np.quantile(citation_counts, 0.5)
    bottom_threshold = np.quantile(citation_counts, 0.25)

    with open(output_path, 'w') as f_out:
        json.dump({
            "top25": top_threshold,
            "middle50": middle_threshold,
            "bottom25": bottom_threshold
        }, f_out, indent=4)
        f_out.write("\n")
            


def compute_publication_types(source_path, save_path):
    with open(source_path, 'r') as f_in:
        publication_types = {}
        for i, line in tqdm(enumerate(f_in), total=4700000, desc="Counting publication types..."):
            record = json.loads(line)
            try:
                for publicationtype in record["publicationtype"]:
                    if publicationtype not in publication_types:
                        publication_types[publicationtype] = 0
                    publication_types[publicationtype] += 1
            except:
                continue
    
    # sort by value
    publication_types = {k: v for k, v in sorted(publication_types.items(), key=lambda item: item[1], reverse=True)}

    with open(save_path, 'w') as f_out:
        json.dump(publication_types, f_out, indent=4)


def count_venues(source_path, save_path):
    with open(source_path, 'r') as f_in:
        venue_counts = {}
        for i, line in tqdm(enumerate(f_in), total=4700000, desc="Counting venues..."):
            record = json.loads(line)
            try:
                if record["venue"] not in venue_counts:
                    venue_counts[record["venue"]] = 0
                venue_counts[record["venue"]] += 1
            except:
                continue

    # sort by value
    venue_counts = {k: v for k, v in sorted(venue_counts.items(), key=lambda item: item[1], reverse=True)}

    with open(save_path, 'w') as f_out:
        json.dump(venue_counts, f_out, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="PubMedCentral file, enriched of MeSH tags and Publication Types and pre-processed (after running both augment and process).")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.")
    parser.add_argument(
        "--citation_thresholds",
        type=str,
        required=True,
        help="JSON file containing the thresholds for the citation count. If not present, they will be computed from the input file.")
    parser.add_argument(
        "--upfold_number",
        type=int,
        default=1,
        help="Number by which all factors are multiplied.")
    parser.add_argument(
        "--mode",
        choices={"filter", "upsample"},
        required=True
    )
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print("Output directory already exists. Do you really want to continue and replace all existing files? (y/n)...")
        choice = input().lower()
        if choice != "y":
            print("Exiting...")
            sys.exit()
        
        # remove all files in output directory
        for filename in os.listdir(args.output_dir):
            os.remove(args.output_dir + filename)

    if not os.path.exists(args.citation_thresholds):
        print("Computing citation thresholds...")
        compute_citation_thresholds(args.input_path, args.citation_thresholds)

    with open(args.citation_thresholds, 'r') as f_in:
        citation_thresholds = json.load(f_in)

    # create log file
    log_path = args.output_dir + "log.txt"
    with open(log_path, 'w') as f_out:
        f_out.write(f"Date: {datetime.now()}\n")
        f_out.write(f"Input path: {args.input_path}\n")
        f_out.write(f"Output directory: {args.output_dir}\n")
        f_out.write(f"Citation thresholds: {args.citation_thresholds}\n")
        f_out.write(f"Upfold number: {args.upfold_number}\n")
        f_out.write(f"Mode: {args.mode}\n")
        f_out.write("\n\n")

    if args.mode == "filter":
        filter(args.input_path, args.output_dir, citation_thresholds, verbose=1)
    elif args.mode == "upsample":
        upsample(args.input_path, args.output_dir, citation_thresholds, upfold=args.upfold_number)
    else:
        raise NotImplementedError("Mode not implemented.")   

    with open(log_path, 'a') as f_out:
        # f_out.write("--------------------------PRE-UPSAMPLING STATISTICS--------------------------\n")
        # compute_statistics(args.input_path, log_path,citation_thresholds)
        # f_out.write("--------------------------POST-UPSAMPLING STATISTICS--------------------------\n")
        compute_statistics(args.output_dir + "train_upsampled.jsonl", log_path, citation_thresholds)
    
    

if __name__ == "__main__":
    main()
