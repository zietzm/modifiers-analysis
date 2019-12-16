import sys
import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
% matplotlib inline
from gensim.models import Word2Vec
import nltk


def clean_string_list(input_list, list_output = False, chars_to_ignore = None, min_char_count = None) :
    output_list = []
    
    for s in input_list:
        if chars_to_ignore :
            for char in chars_to_ignore :
                s = s.replace(char, '  ')
        s = s.split(' ')
        s = [str(x).lower() for x in s]
        s = [str(x).strip() for x in s]
        s = [x for x in s if (x != '-')]
        if min_char_count : s = [x for x in s if (len(x) >= min_char_count)]
        s = [x for x in s if not (x.isdigit())]
        s = [x for x in s if x]
        if list_output == False : s = ' '.join(s)
        output_list.append(s)
        
    return output_list

def get_dataframes_from_csv() :
    # Dump extracted_parents.csv into pandas DataFrame
    extracted = pd.read_csv('extracted_parents.csv', sep = ',')
    # Remove missing criteria strings in df
    extracted = extracted.dropna(how = 'any')
    # Drop duplicates
    extracted = extracted.drop_duplicates()
    # Reset index after drop
    extracted = extracted.reset_index(drop = True)

    # REMOVE ROWS WITH CRITERIA STRINGS THAT DO NOT INCLUDE MATCHED STRING
    indices = [] ; matches = [] ; criteria = []
    for idx, criteria_string in enumerate(list(extracted['criteria_string'])) :

        if (extracted.loc[idx]['matched_string'] in criteria_string) == False :
            indices.append(idx)
            matches.append(extracted.loc[idx]['matched_string'])
            criteria.append(criteria_string)

    # print('Number rows missing relevant matched_string value: {}'.format(len(indices)))
    extracted = extracted.drop(indices)
    extracted = extracted.reset_index(drop = True)
    confirmation_bool = (extracted.shape[0] == (292320 - 62395))

    # NORMALIZE WHITESPACE AND LOWER ALL CASES IN CRITERIA STRING
    cleaned_criteria = clean_string_list(list(extracted['criteria_string']))
    extracted.drop('criteria_string', axis = 1, inplace = True)
    extracted['criteria_string'] = cleaned_criteria

    # SET FILE PATHS
    FILE_ROOT_PATH = 'computed/'
    CONCEPTS_PATH = FILE_ROOT_PATH + 'concepts_with_modifiers.tsv'
    PARENT_PATH = FILE_ROOT_PATH + 'parent_to_descendant_synonyms.tsv'
    PARENT_SYNONYM_PATH = FILE_ROOT_PATH + 'parents_synonyms.tsv'
    PARENT_MODIFIED_CHILDREN_PATH = FILE_ROOT_PATH + 'parents_with_modified_children.tsv'

    # DUMP RELEVANT FILES TO DATAFRAMES
    concepts = pd.read_csv(CONCEPTS_PATH, sep = '\t')
    parents = pd.read_csv(PARENT_PATH, sep = '\t')
    synonyms = pd.read_csv(PARENT_SYNONYM_PATH, sep = '\t')
    modified = pd.read_csv(PARENT_MODIFIED_CHILDREN_PATH, sep = '\t')

    # REMOVE DUPLICATES FROM DATAFRAMES
    concepts = concepts.drop_duplicates()
    concepts = concepts.reset_index(drop = True)
    parents = parents.drop_duplicates()
    parents = parents.reset_index(drop = True)
    synonyms = synonyms.drop_duplicates()
    synonyms = synonyms.dropna(how = 'all')
    synonyms.reset_index(drop = True)
    modified = modified.drop_duplicates()
    modified = modified.dropna(how = 'any')
    modified = modified.reset_index(drop = True)
    
    return extracted, concepts, parents, synonyms, modified

# Function to split top n_rows into the validation set
def split_train_val(n_rows, save = 0) :
    # Top 100 rows serve as validation set
    validation_extracted = extracted.head(n = n_rows).copy()
    # Remaining rows serve as training set
    train_extracted = extracted.tail(n = extracted.shape[0] - (n_rows + 1)).copy()
    # Reset index after drop
    train_extracted = train_extracted.reset_index(drop = True)
    if save == 1 :
        # Export validation df to .csv file
        validation_extracted.to_csv('validation_extracted_parents.csv', encoding = 'utf-8', index = False)
        # Export training df to .csv file
        train_extracted.to_csv('train_extracted_parents.csv', encoding = 'utf-8', index = False)
        
    return train_extracted, validation_extracted

# FUNCTIONS TO QUERY AND NAVIGATE DATAFRAMES
def get_synonyms(parent) :
    parent_list = list(synonyms['parent_concept_name'])
    id_list = list(synonyms['parent_concept_id'])
    
    if parent in parent_list :
        return_df = synonyms[synonyms['parent_concept_name'] == parent]
        return list(return_df['concept_synonym_name'])
    elif parent in id_list :
        return_df =  synonyms[synonyms['parent_concept_id'] == parent]
        return list(return_df['concept_synonym_name'])
    else :
        parent = str(parent)
        parent_list_lower = [str(x).lower() for x in parent_list]
        
        if parent.lower() in parent_list_lower :
            idx = parent_list_lower.index(parent.lower())
            return_df = synonyms[synonyms['parent_concept_name'] == parent_list[idx]]
            return list(return_df['concept_synonym_name'])
        else :
            print('Parent concept {} does not have synonyms.'.format(parent))
            return None
        
def get_parent_concept_id(parent) :
    parent_list = list(synonyms['parent_concept_name'])
    
    if parent in parent_list :
        return list(synonyms[synonyms['parent_concept_name'] == parent]['parent_concept_id'])[0]
    else :
        parent_list_lower = [str(x).lower() for x in parent_list]
        
        if parent.lower() in parent_list_lower :
            idx = parent_list_lower.index(parent.lower())
            return list(synonyms[synonyms['parent_concept_name'] == parent_list[idx]]['parent_concept_id'])[0]
        else :
            print('Parent concept {} not found.'.format(parent))
            return None
        
def get_train_criteria(train) :
    train_criteria = []
    for s in train['criteria_string'] :
        chars_to_ignore = [',', '.', ':', ';', '(', ')', '[', ']', '#', '%', '<', '>', '/', '"', '*', '-', '―']
        s = s.split(' ')
        s = clean_string_list(s, list_output = False, chars_to_ignore = chars_to_ignore)
        train_criteria.append(s)
        
    return train_criteria

def get_embeddings_model(split) :
    train, val = split_train_val(split)
    train_criteria = get_train_criteria(train)
    
    return train, val, Word2Vec(train_criteria, min_count = 1)

def get_closest_string(query, sentence, model, return_index = 0) :
    max_sim = -10
    most_similar = None
    
    if model == None : model = Word2Vec(train_criteria, min_count=1)

    for word in sentence :
        simularity = model.wv.similarity(query, word)
        
        if max_sim < simularity :
            max_sim = simularity
            most_similar = word
            
    if return_index == 0 :
        return most_similar
    else :
        return sentence.index(word)
    
    
# INPUT:
# pandas.DataFrame WITH COLUMNS'criteria_string', 'matched_string', 'parent_concept_id'

# OUTPUT:
# return_ids: LIST OF OMOP CDM CODES ASSOCIATED WITH DESCENDANTS TO GIVEN parent_concept_id
# return_names: LIST OF DESCENDANTS TO GIVEN parent_concept_id
# matches: LIST OF BEST STRING MATCHES TO manual_string GIVEN matched_string
# match_indices: LIST OF FIRST INDEX OF MATCHED STRING IN CLEANED criteria_string

def get_indices_and_potetial_synonyms(df, window = 0, back = False, threshold = 10) :
    
    return_ids = []
    return_names = []
    matches = []
    match_indices = []
    closest_descendents = []
    model = None
    
    required_columns = set(['criteria_string', 'matched_string', 'parent_concept_id'])
    
    if required_columns.issubset(set(df.columns)) :

        detected_string = []
    
        for i, query in enumerate(list(df['criteria_string'])) :
            
            chars_to_ignore = [',', '.', ':', ';', '(', ')', '[', ']', '#', '%', '<', '>', '/', '"', '*', '-', '―']
            query = query.split(' ')
            query = clean_string_list(query, chars_to_ignore = chars_to_ignore)
            
            parent_concept_id = list(df['parent_concept_id'])[i]
            match_candidates = get_descendents_df(parent_concept_id)
            candidate_ids = set(match_candidates['descendant_concept_id'])
            candidate_names = set(match_candidates['descendant_synonym_name'])
            return_names.append(candidate_names)
            return_ids.append(candidate_ids)    

            candidate_words = list(df['matched_string'])[i].split(' ')
            match = [x for x in query if candidate_words[0] in x]
            match_index = 100
            if match and len(candidate_words) == 1 :
                matches.append(match[0])
                match_index = query.index(match[0])
                match_indices.append(match_index)
            elif match and len(candidate_words) > 1 :
                match_index = query.index(match[0])
                match_indices.append(match_index)
                matches.append(' '.join(query[match_index:match_index+len(candidate_words)]))
            elif len(candidate_words) > 1 :
                match = [x for x in query if candidate_words[1] in x]
                match_index = query.index(match[0])
                matches.append(match[0])
                match_indices.append(match_index)
            else :
                match_index = get_index_of_closest(candidate_words[0], query)
                if match_index is None :
                    print('Term {} was not found in criteria string.'.format(' '.join(candidate_words)))
                elif match_index > threshold :
                    match_index = get_closest_string(matched_modifier, s, model = model, return_index = 1)
                    match_indices.append(match_index)
                else :
                    if len(candidate_words) == 1 :
                        match_indices.append(match_index)
                        match = query[match_index]
                        matches.append(match)
                    else :
                        match_indices.append(match_index)
                        match = ' '.join(query[match_index:match_index+len(candidate_words)])
                        matches.append(match)
                        
            if window == 0 :
                search_string = match[0]
            elif match_index - window > 0 and match_index + window < len(query) and back == True :
                search_string = query[match_index - window : match_index + window]
            elif match_index - window > 0 :
                search_string = query[match_index - window : match_index]
            elif back == True :
                search_string = query[match_index : match_index + window]
            else :
                search_string = match[0]
            
            closest_descendent = get_closest_in_list(search_string, list(match_candidates['descendant_synonym_name']))
            closest_descendents.append(closest_descendent)          
            
    
    return matches, match_indices, closest_descendents


def check_modifiers(df, window = 3, severity_modifiers = None, threshold = 100) :
    
    matches, match_indices, closest_descendents = get_indices_and_potetial_synonyms(df,
                                                                                    window = 0,
                                                                                    back = True,
                                                                                    threshold = threshold)
    required_columns = set(['criteria_string', 'matched_string', 'parent_concept_name'])
    
    if required_columns.issubset(set(df.columns)) :
        
        modified = []
        detected_string = []
    
        for idx, s in enumerate(list(df['criteria_string'])) :
            # CLEAN criteria_string
            chars_to_ignore = [',', '.', ':', ';', '(', ')', '[', ']',
                               '#', '%', '<', '>', '/', '"', '*', '-', '―']
            s = s.split(' ')
            s = clean_string_list(s, list_output = False, chars_to_ignore = chars_to_ignore)

            if severity_modifiers == None : severity_modifiers = set(['severe', 'significant', 'major'])
            else :
                severity_modifiers = list(severity_modifiers)
                severity_modifiers = set([str(x).lower() for x in severity_modifiers])

            # CHECK IF PREDEFINED SEVERITY MODIFIERS ARE IN criteria_string
            matched_modifier_set = severity_modifiers.intersection(set(s))
            if len(matched_modifier_set) >= 1:
                matched_modifier = list(matched_modifier_set)[0]
                modifier_index = s.index(matched_modifier)

                # CHECK EXACT STRING MATCH BETWEEN matched_string AND criteria_string
                matched_string = list(df['matched_string'])[idx].split(' ')
                parent_concept_name = list(df['parent_concept_name'])[idx].split(' ')
                parent_concept_name = set([str(x).lower() for x in parent_concept_name])
                concept_matched_string_set = set(parent_concept_name).union(set(matched_string))
                concept_matched_string_set = set(s).intersection(concept_matched_string_set)

                if len(concept_matched_string_set) >= 1 :
                    concept_match = list(concept_matched_string_set)[0]
                    concept_index = s.index(concept_match)

                    # MODIFIER IN FRONT OF CONCEPT
                    if concept_index >= modifier_index and abs(concept_index-modifier_index) < window:
                        modified.append(1)
                        detected_string.append(' '.join(s[modifier_index:concept_index+1]))
                    else :
                        modified.append(0)
                        detected_string.append('')

                else:
                    if set(s).issubset(set(model.wv.vocab)) :
                        closest_index = get_closest_string(matched_modifier, s, model = model, return_index = 1)
                        if closest_index >= modifier_index and abs(closest_index-modifier_index) < window:
                            modified.append(1)
                            detected_string.append(' '.join(s[modifier_index:concept_index+1]))
                        else :
                            modified.append(0)
                            detected_string.append('')
                    else :
                        modified.append(0)
                        detected_string.append('')

            else :
                modified.append(0)
                detected_string.append('')
                
        temp = []
        for x in df['severity_modifier'] :
            if x != 'unmarked' :
                temp.append(1)
            else :
                temp.append(0)
                
        df['truth_label'] = temp
                
        if 'modified' in df.columns : df.drop('modified', axis = 1)
        df['MODIFIED_GUESS'] = modified
        df['CONCEPT_GUESS'] = matches
        df['DESCENDANT_GUESS'] = closest_descendents
                
        return df

    
    else :
        print('Check if input DataFrame columns contain {}'.format(required_columns))
        return None

def get_parent_concept_id(parent) :
    if parent in list(hr['parent_concept_name']) :
        return hr[hr['parent_concept_name'] == parent]['parent_concept_id'][0]
    else :
        parent = str(parent).lower()
        parent_list = list(hr['parent_concept_name'])
        parent_list_lower = [str(x).lower() for x in parent_list]

        if parent in parent_list_lower :
            idx = parent_list_lower.index(parent.lower())
            return hr[hr['parent_concept_name'] == list(hr['parent_concept_name'])[idx]]['parent_concept_id'][0]
        else :
            print('Parent concept {} does not found.'.format(parent))
            return None

def get_descendents_df(parent, return_id = 1) :
    if parent in list(parents['parent_concept_id']) :
        return parents[parents['parent_concept_id'] == parent]
    elif get_parent_concept_id(parent) != None :
        parent = get_parent_concept_id(parent)
        return parents[parents['parent_concept_id'] == parent]
    else :
        print('Parent concept {} does not found.'.format(parent))
        return None
    
def edit_distance_list(search, search_space) :
    return_list = []
    for s in search_space:
        return_list.append(nltk.edit_distance(s, search))
        
    return return_list

def get_index_of_closest(search, search_space) :
    distances = edit_distance_list(search, search_space)
    return distances.index(min(distances))

def get_closest_in_list(search, search_space) :
    return search_space[get_index_of_closest(search, search_space)]

    
def main() :
    
    if sys.argv : csv_import = sys.argv
    else : csv_import = 'annotate_notes_hr2479.csv'
    extracted, concepts, parents, synonyms, modified = get_dataframes_from_csv()

    df = pd.read_csv(csv_import.head().copy(), sep = ',')
    df = df.dropna(subset=['snomed_id'])
    df = df.reset_index(drop = True)

    df = check_modifiers(df, window = 3, severity_modifiers = None)

    df = df[['NCT_id', 'matched_string', 'criteria_string', 'parent_concept_id',
           'parent_concept_name', 'CONCEPT_GUESS', 'DESCENDANT_GUESS']]

    df.to_csv('method_output.csv', sep = ',')
    
if __name__== "__main__":
    main()
