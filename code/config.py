from pymongo import MongoClient

def get_nutrition_fields():
    attrs = ['carbohydrate','cholesterol','fat','fiber','protein','saturatedFat','sodium','sugar','transFat','unsaturatedFat']
    attrs = [c+"Content" for c in attrs]
    attrs = ["calories"] + attrs
    return attrs
    
def get_config():
    import os
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini'))

    return config

def load_ex_controller():
    config = get_config()
    import sys,os
    sys.path.append(config.get('controller','loc'))

def get_mongo_client(connect=True):


    config = get_config()
    client = MongoClient(config.get('mongodb', 'host'),connect=connect)
    return client

def get_mongo_db(connect=True):
    config = get_config()
    client = get_mongo_client(connect=connect)
    return client[config.get('mongodb', 'db')]