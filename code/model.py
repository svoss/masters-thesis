import chainer
import chainer.links as L
import chainer.functions as F
from chainer.serializers import load_npz

""" Models generally consist of two parts:
IngredientSentence: The 1d convolutional layers that maps per ingredient sentence, to a higher level feature array.
Recipe: Combines all the sentence vectors to one general model
Please note that we do not want to use 2d conv layers, altough this is common in image recognition. Since this only makes
sense in a spatial sense as is the case for images. In our case ingredients will be independent.
"""

class BaseIngredientSentence(chainer.Chain):
    """ Defines couple of attribitues that every IngredientSentence should have such that it 
    
    """
    def __init__(self, train, alphabet_size, out_feature,**kwargs):
        self.train = train
        self.alphabet_size = alphabet_size
        print alphabet_size
        self.out_feature = out_feature
        super(BaseIngredientSentence, self).__init__(**kwargs)

class IngredientSentenceSixLayers(BaseIngredientSentence):
    def __init__(self, train, alphabet_size, feature_size=256):
        self.feature_size = feature_size
        super(IngredientSentenceSixLayers, self).__init__(
            train=train,
            alphabet_size=alphabet_size,
            out_feature=self.feature_size,
            conv0=L.Convolution2D(alphabet_size, self.feature_size, (1, 7), stride=1, pad=0),
            conv1=L.Convolution2D(self.feature_size, self.feature_size, (1, 5), stride=1, pad=0),
            conv2=L.Convolution2D(self.feature_size, self.feature_size, (1, 1), stride=1, pad=0),
            conv3=L.Convolution2D(self.feature_size, self.feature_size, (1, 3), stride=1, pad=0),
            conv4=L.Convolution2D(self.feature_size, self.feature_size, (1, 1), stride=1, pad=0),
            conv5=L.Convolution2D(self.feature_size, self.feature_size, (1, 1), stride=1, pad=0),
        )

        # We want to use 1d max-pooling, such that ingredient information is not crossed and stays in tact.
        # The default Functions.max_pooling_nd() function will
        # always take the full input(-2) dimensions of the x input.
        #  In our case this would mean that it would be 2d. Therefore we have to use the class manually.
        # Since our dimensionality  be 4 : BATCH_SIZE X RECIPE_SIZE X TEXT_SIZE X FEATURE_SIZE
        self.maxp3 = F.MaxPooling2D((1,3)) # We will only use max pooling with kernel size 3
        self.maxp2 = F.MaxPooling2D((1,2))  # We will only use max pooling with kernel size 3

    def __call__(self, x):

        # R=Recipe size
        # A=Alphabet size
        # F=Feature size
        # T=text_size
        # 32xAx128
        h = F.relu(self.conv0(x))

        # 32x256x122
        h = self.maxp3(h)
        # 32x256x41
        h = F.relu(self.conv1(h))
        # 32x256x37
        h = self.maxp2(h)
        # 32x256x19
        h = F.relu(self.conv2(h))
        # 32x256x19
        h = F.relu(self.conv3(h))
        # 32x256x17
        h = F.relu(self.conv4(h))
        # 32x256x17
        h = F.relu(self.conv5(h))
        # 32x256x17
        h = self.maxp3(h)
        # 32x256x6
        return h


class BaseRecipe(chainer.Chain):
    def __init__(self, train, recipe_size, ingredient, **kwargs):
        super(BaseRecipe, self).__init__(ingredient=ingredient,**kwargs)
        self.train = train
        self.recipe_size = recipe_size

    """ Define train as property such that it can be passed to recipe
    """
    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val
        self.ingredient.train = val



class FCRecipeCuisine(BaseRecipe):
    def __init__(self, train, recipe_size, ingredient):
        self.output_size = output_size

        super(FCRecipeCuisine, self).__init__(
            train=train,
            recipe_size=recipe_size,
            ingredient=ingredient,
            fc1=L.Linear(49152, 1024),
            fc2=L.Linear(1024, 1024),
            fc3=L.Linear(1024, self.output_size)
        )

    def __call__(self, x):
        h = self.ingredient(x)

        h = F.dropout(F.relu(self.fc1(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc2(h)), train=self.train, ratio=0.5)
        return self.fc3(h)

class MaxRecipeCuisine(BaseRecipe):
    def __init__(self, train, recipe_size, ingredient, output_size):
        self.output_size = output_size

        super(MaxRecipeCuisine, self).__init__(
            train=train,
            recipe_size=recipe_size,
            ingredient=ingredient,
            fc1=L.Linear(1536, 1024),
            fc2=L.Linear(1024, 1024),
            fc3=L.Linear(1024, self.output_size)
        )

        self.maxp6 = F.MaxPooling2D((recipe_size, 1))

    def __call__(self, x):
        h = self.ingredient(x)
        h = self.maxp6(h)
        h = F.dropout(F.relu(self.fc1(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc2(h)), train=self.train, ratio=0.5)
        return self.fc3(h)

def build_model(alphabet,  output_size, recipe_size=32, recipe_type='fc',ing_type='6l-256',load_file=None):

    if ing_type == '6l-256':
        ing = IngredientSentenceSixLayers(True, len(alphabet))
    else:
        raise Exception("ingredient type: %s not supported " % ing_type)


    if recipe_type == 'max':
        recipe = MaxRecipeCuisine(True, recipe_size, ing, output_size)
    elif recipe_type == 'fc':
        recipe = FCRecipeCuisine(True, recipe_size, ing, output_size)
    else:
        raise Exception("recipe type: %s not supported " % ing_type)
    if load_file is not None:
        load_npz(load_file, recipe)
    return recipe