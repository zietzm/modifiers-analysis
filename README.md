# Map-Modifiers

## Data in this repository

`data/` contains one file and one subdirectory.
`data/extracted_parents.csv.xz` is a compressed `.csv` file, which has the following columns:

```
NCT_id, matched_string, criteria_string, parent_concept_id, parent_concept_name
```

The meaning of these columns is as follows:

* `NCT_id` - the inclusion/exclusion criteria for the trial. Each `NCD_id` can appear multiple times.
* `matched_string` - the string that was matched in the criteria. These are synonyms of parent concepts.
* `criteria_string` - the string in which the `matched_string` was matched. These are 100 characters on either side of the start of the `matched_string`.
* `parent_concept_id` - the ID of the concept corresponding to `matched_string`
* `parent_concept_name` -  the preferred name corresponding to the `parent_concept_id`, of which `matched_string` is a synonym.

Within the subdirectory, `data/computed`, there are the following files:

* `concept_with_modifiers.tsv` - Concepts that have one of significant, severe, or serious modifiers.
* `parents_with_modified_children.tsv` - The immediate parents of concepts in `concepts_with_modifiers.tsv` and those relationships.
* `parent_synonyms` - Synonyms of the parent concepts from `parents_with_modified_children.tsv` (from SNOMED)
* `parent_to_descendant_synonyms` - A map between parent concepts and the synonyms of their children. This is used for extracting the modified (or not) concepts once a parent concept has been found. This file is particularly important for Young's part, where we need to go from a parent concept match to a child concept (or the concept itself).


## Data sources:

* ClinicalTrials.gov data
    * https://clinicaltrials.gov/ct2/resources/download#DownloadAllData
* SNOMED CT
    * ATHENA - http://athena.ohdsi.org/vocabulary/list
