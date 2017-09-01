from dataset import \
    download_if_not_exists,create_sub_dataset, get_cache_folder, create_alphabet, map_word_sequence, find_chars, apply_threshold_to_voc, create_normal_voc, convert_to_count_array
import os
import json
import hashlib
import numpy as np

VEGAN = ['vegetarian', 'vegan']
MEAT = ['fish', 'chicken', 'pork', 'beef', 'pork recipes', 'lamb', 'beef recipes', 'duck', 'duck recipes', 'turkey', 'turkey recipes', 'seafood recipes', 'seafood', 'lamb recipes']
CUISINES = ['italian', 'asian', 'american', 'french', 'indian', 'british', 'mexican', 'chinese', 'mediterranean']

PRELIMINARY_DATASET = 'http://datasets.stijnvoss.com/recipes/preliminary.zip'

def _determine_labels(recipe, veg_tags, meat_tags, cus_tags):
    """ Helper function that determines cuisine and vegetarian label for a recipe
    """
    veg = 0
    cus = 0
    for t in _extract_tags(recipe):
        t = t.lower()
        if t in veg_tags:
            veg = 1
        elif t in meat_tags:
            veg = 2
        if t in cus_tags:
            cus = cus_tags.index(t) + 1
    return veg,cus


def _extract_tags(recipe):
    if type(recipe['tags']) is not list:
        recipe['tags'] = [recipe['tags']]
    for tag in recipe['tags']:
        if type(tag) is not list:
            yield tag
        else:
            for t in tag:
                yield t


def _read_recipes(dataset_url):
    folder = download_if_not_exists(dataset_url)
    for filename in os.listdir(folder):
        json_file = os.path.join(folder, filename)
        if os.path.isfile(json_file) and filename[-5:] == '.json':
            with open(json_file,'r') as io:
                data = json.load(io)
                for recipe in data['recipes']:
                    yield recipe


def _map_to_new_voc(Y, mapping):
    """Maps from old voc to new voc recusivly in case you want to rewrite your vocu"""
    if type(Y) is list:
        return [x for x in [_map_to_new_voc(y,mapping) for y in Y] if x is not None]
    elif Y in mapping:
        return mapping[Y]
    else:
        return None


def get_preliminary_dataset(dataset_url, case_sensitive=True, digits=True, others="%.&[]()!~-+ ", cusines=CUISINES,
                  vegetarian=VEGAN, meat=MEAT, name_th=5, max_text_size=128, max_recipe_size=32,test=False):
    function_str = ".".join([str(v) for k,v in locals().iteritems()])

    npz_file = os.path.join(get_cache_folder(), hashlib.md5(function_str).hexdigest() + ".npz")
    if not os.path.exists(npz_file):
        char2int, int2char = create_alphabet(case_sensitive, digits, others)
        if len(int2char) > 255:
            raise Exception("Alphabet size of %d exceeds maximum of 255 characters")
        X_char = []
        X_words = []
        voc_ingredients = {}
        voc_names = {}
        voc_names_counter = {}

        Y_cuisine = []
        Y_vegetarian = []
        Y_names = []
        counter = 0
        LIMIT = 1000
        for recipe in _read_recipes(dataset_url):
            if recipe['recipe_name'] is None:
                continue

            # name
            names = map_word_sequence(recipe['recipe_name'], voc_names, voc_names_counter)

            # Ingredients:
            chars = []
            words = []
            if len(recipe['ingredients']) > max_recipe_size:
                continue

            for ingredient in recipe['ingredients']:
                # chars
                l = len(ingredient)
                if l > max_text_size:
                    continue
                if not case_sensitive:
                    ingredient.lower()
                chars.append(find_chars(ingredient, l, max_text_size, char2int))
                words.extend(map_word_sequence(ingredient, voc_ingredients))

            for i in xrange(len(chars), max_recipe_size):
                chars.append([0 for i in xrange(max_text_size)])

            X_char.append(chars)
            X_words.append(words)
            Y_names.append(names)

            # Label:
            veg, cus = _determine_labels(recipe, vegetarian, meat, cusines)
            Y_cuisine.append(cus)
            Y_vegetarian.append(veg)
            counter += 1

        # This removes names of recipes that are below the threshold and creates new vocabulary
        Y_names, voc_names = apply_threshold_to_voc(Y_names, voc_names, voc_names_counter, name_th)
        # While building this dataet it was easier to used in the inversed vocubalaries,
        # But for numpy we are going to need lists..
        voc_names = create_normal_voc(voc_names)
        voc_ingredients = create_normal_voc(voc_ingredients)
        # convert list of characters manually, also make sure that 0 character is present
        int2char[0] = ''
        char_voc = [int2char[i] for i in xrange(len(int2char))]

        # Lastly we are going to convert everything to numpy before we save it:
        X_char = np.array(X_char, dtype=np.int8)
        X_words = convert_to_count_array(X_words, dtype=np.int8)
        Y_names = convert_to_count_array(Y_names, dtype=np.int8)
        Y_cuisine = np.array(Y_cuisine, dtype=np.int8)
        Y_vegetarian = np.array(Y_vegetarian, dtype=np.int8)

        if test:
            X_char = X_char[:1000]
            X_words = X_words[:1000]
            Y_names = Y_names[:1000]
            Y_cuisine = Y_cuisine[:1000]
            Y_vegetarian = Y_vegetarian[:1000]

        int2char = np.array(char_voc, dtype=np.unicode_)
        voc_ingredients = np.array(voc_ingredients, dtype=np.unicode_)
        voc_names = np.array(voc_names, dtype=np.unicode_)

        # Save it
        np.savez(npz_file, X_char=X_char, X_words=X_words, Y_names=Y_names, Y_cuisine=Y_cuisine,
                 Y_vegetarian=Y_vegetarian, int2char=int2char, voc_ingredients=voc_ingredients, voc_names=voc_names)
    else:
        data = np.load(npz_file)

        X_char, X_words = data['X_char'], data['X_words']
        Y_names, Y_cuisine, Y_vegetarian = data['Y_names'], data['Y_cuisine'], data['Y_vegetarian']
        int2char, voc_ingredients, voc_names = data['int2char'], data['voc_ingredients'], data['voc_names']
    import humanize
    size = humanize.naturalsize(os.path.getsize(npz_file))
    print "Using dataset of %s. Found a total of %d recipes" % (size, X_char.shape[0])
    return (X_char, X_words), (Y_names, Y_cuisine, Y_vegetarian), (int2char, voc_ingredients, voc_names)


def get_dataset(n_chars, case_sensitive, ds_parts='cuisine', max_recipe_size=32, test_mode=False,val_chars=None):

    X, Y, voc = get_preliminary_dataset(PRELIMINARY_DATASET, name_th=5, max_text_size=n_chars,max_recipe_size=max_recipe_size,
                                        case_sensitive=case_sensitive,test=test_mode)
    Y_cuisine, Y_veg = Y[1], Y[2]
    Y_cuisine =  np.array([0 if y == 0 else (1 if y == 3 else 2) for y in Y_cuisine.tolist()], dtype=np.int32)
    X_chars = X[0]
    alphabet = voc[0]
    if ds_parts in ['union', 'intersect']:
        Y = zip(Y_cuisine, Y_veg)
        if ds_parts == 'union':
            X,Y = create_sub_dataset(X_chars, Y, lambda y: (y[0] + y[1]) > 0, lambda y: (y[0]-1, y[1]-1))

            return X, Y.astype(np.int32), alphabet
        else:
            X,Y = create_sub_dataset(X_chars, Y, lambda y: y[0] > 0 and y[1] > 0, lambda y: (y[0]-1, y[1]-1))
            return X, Y.astype(np.int32), alphabet
    elif ds_parts == 'cuisine':
        X, Y = create_sub_dataset(X_chars, Y_cuisine, lambda y: y > 0, lambda y: y - 1)
        Y = Y.astype(np.int32)
        return X,Y,alphabet
    elif ds_parts == 'vegatarian':
        X, Y = create_sub_dataset(X_chars, Y_veg, lambda y: y > 0, lambda y: y - 1)
        Y = Y.astype(np.int32)
        return X, Y, alphabet

    raise Exception("Unkown dataset variant %s " % ds_parts)