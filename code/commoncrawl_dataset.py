from dataset import MongoDBDataset, create_alphabet, find_chars, build_mongo_datasets
from config import get_nutrition_fields,get_mongo_db
import chainer.datasets as D 
from chainer import dataset
import numpy as np
import math
import operator
DEFAULT_RY_CLASSES = [(0,1,1),(2,3,2),(4,4,4),(5,6,6),(7,8,8),(9,11,10),(12,14,12),(15,19,16),(20,29,22),(30,72,42)]
class BaseYEncoder(object):
    def _get_fields(self, recipe):
        return [self._get_field(recipe,f) for f in self.fields]

    def _get_field(self, recipe, f):
        atts = f.split(".")
        for att in atts:
            if att not in recipe:
                return None
            recipe = recipe[att]
        return recipe

    def __call__(self, recipes):
        raise NotImplementedError()

class YContinuesEncoder(BaseYEncoder):
    def __init__(self, fields, add_recipe_yield=False, multiply_with_recipe_yield = True):
        self.fields = fields
        self.add_recipe_yield = add_recipe_yield
        self.multiply_with_recipe_yield = multiply_with_recipe_yield

    def __call__(self, recipes):
        multiple = type(recipes) is list
        if not multiple:
            recipes = [recipes]
        Y = []
        for recipe in recipes:
            y = self._get_fields(recipe)
            if self.multiply_with_recipe_yield:
                y = [a * recipe['parsed']['recipe_yield'] * 0.01 for a in y]
            if len(y) < 2:
                Y.append(y[0])
            else:
                Y.append(y)
        if multiple:
            y = np.array(Y, np.float32)
            return y

        else:
            return np.array(Y[0],np.float32)

class YCategorizedEncoder(BaseYEncoder):
    def __init__(self, num_categories, fields, add_recipe_yield=False, multiply_with_recipe_yield=True,ry_classes=None):
        self.fields = fields
        self.add_recipe_yield = add_recipe_yield
        self.multiply_with_recipe_yield = multiply_with_recipe_yield
        self.classes = {}
        self.ry_classes = ry_classes
        self.num_categories = num_categories

    def prepare_classes(self, all_recipes):
        values = dict([(f,[]) for f in self.fields])
        c = 0
        for r in all_recipes:
            c += 1
            recipe_yield = self._get_field(r, 'parsed.recipe_yield')
            for f in self.fields:
                x = self._get_field(r, f)
                if x is not None:
                    values[f].append(x * recipe_yield)
        self.classes = dict([(f,self._build_classes(L)) for f,L in values.iteritems()])
        if self.add_recipe_yield:
            self.classes['parsed.recipe_yield'] = self.ry_classes
        print self.classes

    def _build_classes(self, values):
        values = sorted(values)
        classes = []
        for c in self._chunks(values, int(math.ceil(float(len(values))/self.num_categories))):
            middle = len(c)//2
            classes.append([c[0], c[-1], c[middle]])
        return classes


    def _chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _search_classes(self, Y):
        y_new = []
        F = self.fields[:]
        if self.add_recipe_yield:
            F.append('parsed.recipe_yield')
        for i,f in enumerate(F):
            y = Y[i]
            classes = self.classes[f]
            for ci,c in enumerate(classes):
                if c[0] > y:
                    ci = ci-1
                    break
            y_new.append(ci)

        return y_new


    def __call__(self, recipes):
        multiple = type(recipes) is list
        if not multiple:
            recipes = [recipes]
        Y = []
        for recipe in recipes:
            y = self._get_fields(recipe)
            ry = recipe['parsed']['recipe_yield']
            if self.multiply_with_recipe_yield:
                y = [-1 if a is None else a * ry for a in y]

            if self.add_recipe_yield:
                y.append(ry)


            y = self._search_classes(y)

            if len(y) < 2:
                Y.append(y[0])
            else:
                Y.append(y)

        if len(Y) > 1:
            print "Multiple",len(Y)
            return np.array(Y, np.int32)
        else:
            return np.array(Y[0],np.int32)

class DocumentEncodingDataset(dataset.DatasetMixin):
    """ Dataset type that takes a document dataset and output integer representations of the ingredients as X and a specified field als Y
    """
    def __init__(self, db,num_chars, num_ingredients, y_encoder):
        self.db = db
        self.char2int,self.int2char = create_alphabet(case_sensitive=False, digits=True, others="/\,%.&[]()!~-+ ",do_unicode=True)
        self.y_encoder = y_encoder
        self.num_chars = num_chars
        self.num_ingredients = num_ingredients

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        """
        If index is slice will return N+1 dimensions, if scalar will return N dimensions. Second dimension will be vector_size
        :param index: 
        :return: 
        """
        if isinstance(index, slice):
            recipes = self._convert_items(self.db[index])
            
        else:
            recipes = self._convert_item(self.db[index])
        return recipes

    def alphabet_size(self):
        return len(self.char2int) + 1

    def _convert_items(self, recipes):
        R = []
        for recipe in recipes:
            R.append(self._convert_item(recipe))

        return R

    def _convert_item(self, recipe):

        x = self._convert_ingredients(recipe['ingredients'])



        return np.array(x,np.int32), self.y_encoder([recipe])

    def _convert_ingredients(self, ingredients):
        X = []
        for idx in xrange(self.num_ingredients):
            ing = ingredients[idx] if len(ingredients) > idx else ""
            ing = ing.lower()
            x = find_chars(ing, len(ing), self.num_chars, self.char2int)
            X.append(x)
        return X

    def print_recipe(self, ingredients):
        _,recipe_size, ingredient_size =  ingredients.shape
        for ing_idx in xrange(recipe_size):
            ing = ingredients[0,ing_idx,:]
            if len(ing) > 0:
                print "".join([self.int2char[ing[c]] if ing[c] > 0 else "" for c in xrange(ingredient_size)])

class XOneHotEncodingDataset(dataset.DatasetMixin):
    """ Class that converts integers to one-hot-encoded vectors. 0's will be encoded as all 0's"""
    def __init__(self, db, vector_size, dtype=np.float32):
        """
        :param X: N-dimesional dataset of integer values
        :param vector_size: alphabet size of(The max integer found in X)
        """
        self.db = db
        self.vector_size = vector_size
        self.dtype = dtype

    def __getitem__(self, index):
        """
        If index is slice will return N+1 dimensions, if scalar will return N dimensions. Second dimension will be vector_size
        :param index: 
        :return: 
        """
        if isinstance(index,slice):
        
            return self._one_hot_encode(self.db[index],slice=True)
        else:
            return self._one_hot_encode(self.db[index],slice=False)

    def _one_hot_encode(self, data,roll_to=0,slice=False):
        D = []
        if not slice:
            data = [data]
        for X,Y in data:
            shape = list(X.shape)

            l = reduce(operator.mul, shape, 1) # Total length if was list of ints
            X = X.reshape((l,)) # reshape it to list of ints
            encoded = np.zeros((l, self.vector_size + 1),self.dtype) # Build matrix of l x vector_size + 1
            encoded[np.arange(l), X] = 1 # One on the right places
            encoded = encoded[:, 1:] # remove first row, 0 = all zero's
            encoded = encoded.reshape(tuple(shape + [self.vector_size])) # reshape back
            encoded = np.rollaxis(encoded, len(shape), roll_to)
            D.append((encoded,Y))
        if slice:
            return D
        return D[0]

    def __len__(self):
        return len(self.db)

def get_fields(ds):
    if ds in get_nutrition_fields():
        return ['parsed.%s' % ds], False, True
    if ds == 'the-five-union':
        fields = ['calories','cholesterolContent', 'proteinContent', 'transFatContent']
        return ['parsed.%s' % d for d in fields],True, True


def get_nutrition_dataset(args):
    query = {"context.language":"en","parsed.recipe_yield":{"$ne":None}, 'max_ingredient_size': {"$lte":args.num_chars},'recipe_size':{"$lte":args.num_ingredients},'recipe_size':{"$gt":0}}
    db = get_mongo_db(False)
    collection = db['recipes']
    fields, add_recipe_yield, union = get_fields(args.dataset)
    if len(fields) == 1 or not(union):
        for ds in fields:
            query[ds] = {"$ne":None}
    else:
        query["$or"] =  [dict([(f,{"$ne":None})])  for f in fields]

    datasets = list(build_mongo_datasets(collection, query, test_frac=0.0, limit_to=(1000 if args.test else args.limit_to)))

    datasets = [MongoDBDataset(collection,x) for x in datasets]

    if args.categories is None:
        encoder = YContinuesEncoder(fields, add_recipe_yield)
    else:
        encoder = YCategorizedEncoder(args.categories,fields,add_recipe_yield,ry_classes=DEFAULT_RY_CLASSES)
        encoder.prepare_classes(collection.find(query))

    datasets = [DocumentEncodingDataset(x, args.num_chars, args.num_ingredients, encoder) for x in datasets]
    alphabet_size = datasets[0].alphabet_size()
    datasets = [XOneHotEncodingDataset(x,alphabet_size) for x in datasets]
    
    train, val, test = tuple(datasets)
    return train, val, test, alphabet_size

if __name__ == "__main__":
    encoder = YCategorizedEncoder(3, ['a'], add_recipe_yield=True, ry_classes=DEFAULT_RY_CLASSES)
    a = [1,1,1,1,1,2,3,4,5,6,7,10,14,18]
    b = [8,8.5,9,9.5,10,90,95,100,105,110,900,950,1000,1100]
    test = [{'a':a[i], 'b':b[i],'parsed':{'recipe_yield':i}} for i in xrange(0,14)]
    encoder.prepare_classes(test)
    print encoder([test[4],test[10],test[5]])
