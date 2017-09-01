import requests
from bs4 import BeautifulSoup
from rdflib import Graph, plugin
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from config import get_mongo_db
import os
def find_correct_recipe_yield(recipes):
    corrections = []
    for url in recipes:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')
        x = soup.find_all("script", type="application/ld+json")
        if len(x) > 0:
            data = json.loads(x[0].contents[0])
            if 'recipeYield' in data:
                corrections.append((url, data['recipeYield']))
    return corrections


corrections = []


def repair_food_com():
    db  = get_mongo_db()
    results = [r['context']['url'] for r in db.recipes.find({"context.host":"www.food.com"})]
    print len(results)
    existing_urls = []
    corrections = []
    if os.path.exists('repair_recipe_yield.csv'):
        df = pd.read_csv('repair_recipe_yield.csv')
        existing_urls = [u.url for u in df.itertuples()]
        corrections = [(u.url, u.recipeYield) for u in df.itertuples()]
    chunks = [[] for i in xrange(10)]
    counter = 0
    bar = tqdm(total=(len(results) / 1000))
    p = Pool(10)
    for r in results:
        if r not in existing_urls:
            counter += 1
            chunks[counter % 10].append(r)
            if counter % 100 == 0:
                m = p.map(find_correct_recipe_yield, chunks)
                for cor in m:
                    corrections.extend(cor)
                df = pd.DataFrame(corrections, columns=["url", "recipeYield"])
                df.to_csv('repair_recipe_yield.csv', encoding='utf8')
                chunks = [[] for i in xrange(10)]
                bar.update()

    m = p.map(find_correct_recipe_yield, chunks)
    for cor in m:
        corrections.extend(cor)
    df = pd.DataFrame(corrections, columns=["url", "recipeYield"])
    df.to_csv('repair_recipe_yield.csv', encoding='utf8')
    print([len(c) for c in chunks])

    # df = pd.DataFrame(corrections,columns=["url","recipeYield"])
    # df.to_csv('repair_recipe_yield.csv',encoding='utf8')"""


repair_food_com()
