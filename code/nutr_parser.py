"""
This script parses the nutrional information and removes outliers.
"""
import argparse
import re
from tqdm import tqdm
import numpy as np
from config import get_config, get_mongo_db, get_mongo_client

# Some pre-defined regular expressions used by recipe yield extractor and serving size extractor, to make sure compiling happens only once
p0 = re.compile("(\d+)(?:-\d+)? serving",re.IGNORECASE)
p1 = re.compile("(\d+)",re.IGNORECASE)
p3 = re.compile("\d+\/\d+",re.IGNORECASE)
p2 = re.compile("\d+\s?g",re.IGNORECASE)

nut0 = re.compile("(\d+)(?: (\d+)\/(\d+))",re.IGNORECASE)
nut1 = re.compile("(\d[\d\.]+)",re.IGNORECASE)

#helper functions
def _make_float(f):
    """ Converts str value to float """
    f = f.strip()
    if f.count('.') < 2:
        return float(f.strip())
    return None

def _add_to_list_if_not_none(d,k,v,ignore_zero=False):
    """ Adds v to d[k] if v is not none
    """
    if k not in d:
        d[k] = []
    if v is not None and not(ignore_zero and v == 0.0):
        d[k].append(v)

def _determine_thresholds(frac, values_parsed):
    """ Determines thresholds for each list in values_parsed dict. First frac and last frac will be below and above thresholds
    """
    thresholds = {}
    for key,value in values_parsed.iteritems():
        L = sorted(value)
        idx = int(frac*len(L)) 
        thresholds[key] = (L[idx-1],L[-idx])
    return thresholds

def _print_th(thresholds):
    """ Print overview of the thresholds used """
    for key, th in thresholds.iteritems():
        print "%s: %.2f-%.2f" % (key, th[0],th[1])
    print "---"

def _add_to_parsed_if_th(D,th,k,v,th_v,ignore_zero=False):
    """ Adds v to D[k], if th_v is within thresholds(th) otherwise will add None """
    if v is not None and not (ignore_zero and v == 0.0):
        if th_v < th[0] or th_v > th[1]:
            v = None
    D[k] = v
    return v is not None

def parse_nutrition(nutr):
    """ Tries to parse a nutrition field will return None if not certain"""
    if nutr is None:
        return None
    match = nut0.search(nutr)
    if match:
        return _make_float(match.group(1)) + _make_float(match.group(2)) / _make_float(match.group(3))
    match = nut1.search(nutr)
    if match:
        return _make_float(match.group(1))
    return None


def parse_yield(recipeYield, servingSize):
    """ Tries to parse a recipe yield field will return None if not certain"""
    if recipeYield is not None:
        if servingSize is None or servingSize.strip() == '' or servingSize.strip().lower()[0] == '1' or servingSize == recipeYield or p2.match(servingSize):
            match = p3.search(recipeYield)
            if match is not None:
                return None
            match = p0.search(recipeYield)
            if match:
                return float(match.group(1))
            match = p1.search(recipeYield)
            if match:
                return float(match.group(1))

    return None

def define_args():
    # Args of this script
    parser = argparse.ArgumentParser(description='This script parses the nutrional information and removes outliers only works on the english language')
    parser.add_argument('--frac-removed', default=.01, type=float, help='Fraction  of outliers removed on each side, based on the total recipe size')
    args = parser.parse_args()
    return args



def parse(db):
    # Walk over all data to make sure 
    b = tqdm(total=db.recipes.count())
    values_parsed = {}
    print "Reading data"
    for r in db.recipes.find():
        # determine recipe_yield first
        serving_size = r['nutritionInfo']['servingSize'] if 'nutritionInfo' in r and 'servingSize' in r['nutritionInfo'] else None
        recipe_yield = parse_yield(r['recipeYield'], serving_size) if 'recipeYield' in r else None
        _add_to_list_if_not_none(values_parsed, 'recipe_yield', recipe_yield)
        if recipe_yield is not None and 'nutritionInfo' in r:
            nutr = r['nutritionInfo']
            for key, value in nutr.iteritems():
                nut = parse_nutrition(value)
                if nut is not None and recipe_yield is not None:
                    # Threshold is based on total recipe nutrional value
                    nut = recipe_yield * nut
                    _add_to_list_if_not_none(values_parsed, key, nut,ignore_zero=True)

        b.update()
    b.close()

    # this determines the thresholds
    TH = _determine_thresholds(args.frac_removed,values_parsed)
    _print_th(TH)
    
    # Only add values that are within thresholds
    print "Updating data with thresholds"
    
    b = tqdm(total=db.recipes.count())
    for r in db.recipes.find():
        d = {}
        serving_size = r['nutritionInfo']['servingSize'] if 'nutritionInfo' in r and 'servingSize' in r['nutritionInfo'] else None
        recipe_yield = parse_yield(r['recipeYield'], serving_size) if 'recipeYield' in r else None
        added = _add_to_parsed_if_th(d, TH['recipe_yield'], 'recipe_yield', recipe_yield, recipe_yield)
        if added and 'nutritionInfo' in r:
            nutr = r['nutritionInfo']
            for key, th in TH.iteritems():
                val = parse_nutrition(nutr[key]) if key in nutr else None
                if val is not None:
                    th_val = val * recipe_yield
                    _add_to_parsed_if_th(d, th, key, val, th_val,ignore_zero=True)
        r['parsed'] = d
        db.recipes.save(r)
        b.update()
    b.close()

if __name__ == '__main__':
    args = define_args()
    db = get_mongo_db()
    parse(db)
