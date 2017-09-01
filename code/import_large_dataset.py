from config import get_config, get_mongo_db, get_mongo_client
import argparse
import json
import os
def build_args():
    parser = argparse.ArgumentParser(description='Takes dataset obtained via spark from common ')
    parser.add_argument('folder', type=str, help='Dir where the jsons are located')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    return args

def iterate_lines(args):
    if not os.path.isdir(args.folder):
        raise Exception("%s is not a dictionary" % args.folder)
    for dirpath, dnames, fnames in os.walk(args.folder):
        for f in fnames:
            if f.startswith("part"):
                print "reading %s" % f
                with open(os.path.join(args.folder, f)) as io:
                    for l in io:
                        yield json.loads(l)

def convert_doc(l):
    l['recipe_size'] = len(l['ingredients'])
    max_chars = 0
    for ingredient in l['ingredients']:
        if ingredient is None:
            ingredient = ""
        if max_chars < len(ingredient):
            max_chars = len(ingredient)
    l['max_ingredient_size'] = max_chars

    # Html context says it's enligsh...
    if l['context']['host'] == 'eda.ru':
        l['context']['language'] = 'ru'
    
    return l


def import_to_db(db, args):
    rows = []
    total = 0
    count = 0
    for l in iterate_lines(args):
        rows.append(convert_doc(l))
        count += 1
        total += 1
        if count > 999:
            count = 0
            db.recipes.insert_many(rows,ordered=False)
            rows = []
        if total % 100000 == 0:
            print total

    if len(rows) > 0:
        db.recipes.insert_many(rows,ordered=False )
    
    print "Inserted %d recipes" % total

def remove_db():
    config = get_config()
    client = get_mongo_client()
    client.drop_database(config.get('mongodb','db'))

if __name__ == '__main__':
    args = build_args()
    if args.refresh:
        remove_db()
    db = get_mongo_db()
    #print  db.recipes.find({"ingredients":[]}).count()
    #for rec in db.recipes.find({"context.language":"ru"}):
    #     print rec['context']['url']
    import_to_db(db, args)