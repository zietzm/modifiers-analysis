import json

import requests

# ! This should move to a new module. It has nothing to do with our method,
# ! only evaluation.

def get_c2q_mapping(string):
    """
    Returns Criteria2Query mappings from a given string.

    Note that request throttling means you should space out calls of this
    function. Otherwise requests will fail because of rate limits.
    """
    post_data = {
        'inc': string,
        'exc': '',
        'initialevent': '',
        'rule': True,
        'ml': True,
        'abb': True,
        'obstart': '',
        'obend': '',
        'daysbefore': '0',
        'daysafter': '0',
        'limitto': 'All'
    }
    base = 'http://www.ohdsi.org/web/criteria2query/'
    with requests.Session() as s:
        # Clear cookies
        s.cookies.clear()

        # Post the data first
        post_response = s.post(
            base + 'main/autoparse',
            data=post_data
        )

        get_response = s.get(
            base + 'queryformulate/formulateCohort',
            data={}
        )

    if get_response.status_code in {500, 504}:
        return post_response, get_response

    responses_list = (
        json.loads(get_response.json()['jsonResult'])
        ['ConceptSets']
    )
    return responses_list


def format_results(results, criteria_dict):
    """Format results depending on whether successful or erroneous"""
    if isinstance(results, tuple):
        return [], _format_error(results, criteria_dict)
    return _format_correct(results, criteria_dict), []


def _format_error(results, criteria_dict):
    criteria_dict.update({
        'post_request': results[0],
        'get_request': results[1],
    })
    return [criteria_dict, ]


def _format_correct(results, criteria_dict):
    outputs = list()
    for res in results:
        if not res.get('name'):
            continue
        result_dict = {
            **criteria_dict.copy(),
            'cohort_name': res['name'],
        }
        for item in res['expression']['items']:
            outputs.append({
                **result_dict,
                'concept': item['concept'],
            })
    return outputs
