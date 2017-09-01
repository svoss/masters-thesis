import chainer
import chainer.links as L
import chainer.functions as F
from chainer.serializers import load_npz
from chainer import reporter
import numpy as np
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
        self.out_feature = out_feature
        super(BaseIngredientSentence, self).__init__(**kwargs)

class IngredientSentenceSixLayers(BaseIngredientSentence):
    def __init__(self, train, alphabet_size, feature_size=256):
        self.feature_size = feature_size
        super(IngredientSentenceSixLayers, self).__init__(
            train=train,
            alphabet_size=alphabet_size,
            out_feature=self.feature_size*6,
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


class IngredientSentenceSmallerLayers(BaseIngredientSentence):
    def __init__(self, train, alphabet_size, feature_size=256):
        self.feature_size = feature_size
        super(IngredientSentenceSmallerLayers, self).__init__(
            train=train,
            alphabet_size=alphabet_size,
            out_feature=self.feature_size*6,
            conv0=L.Convolution2D(alphabet_size, self.feature_size, (1, 2), stride=1, pad=0),
            conv1=L.Convolution2D(self.feature_size, self.feature_size, (1, 3), stride=1, pad=0),
            conv2=L.Convolution2D(self.feature_size, self.feature_size, (1, 3), stride=1, pad=0),
            conv3=L.Convolution2D(self.feature_size, self.feature_size, (1, 3), stride=1, pad=0),
            conv4=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),

            conv5=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),
            conv6=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),

            conv7=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),
            conv8=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),
            conv9=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),

            conv10=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),
            conv11=L.Convolution2D(self.feature_size, self.feature_size, (1, 2), stride=1, pad=0),
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

        # 32x256x127
        h = F.relu(self.conv1(h))
        # 32x256x125
        h = F.relu(self.conv2(h))
        # 32x256x123
        h = F.relu(self.conv3(h))
        # 32x256x121
        h = F.relu(self.conv4(h))
        # 32x256x121
        h = self.maxp3(h)
        # 32x256x40
        h = F.relu(self.conv5(h))
        # 32x256x39
        h = F.relu(self.conv6(h))
        # 32x256x38
        h = self.maxp2(h)
        # 32x256x19
        h = F.relu(self.conv7(h))
        # 32x256x18
        h = F.relu(self.conv8(h))
        # 32x256x17
        h = F.relu(self.conv9(h))
        # 32x256x16
        h = self.maxp2(h)
        # 32x256x8
        h = F.relu(self.conv10(h))
        # 32x256x7
        h = F.relu(self.conv11(h))
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
    def __init__(self, train, recipe_size, ingredient,output_size):
        self.output_size = output_size

        super(FCRecipeCuisine, self).__init__(
            train=train,
            recipe_size=recipe_size,
            ingredient=ingredient,
            fc1=L.Linear(ingredient.out_feature * recipe_size, 1024),
            fc2=L.Linear(1024, 1024),
            fc3=L.Linear(1024, self.output_size)
        )

    def __call__(self, x):
        h = self.ingredient(x)

        h = F.dropout(F.relu(self.fc1(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc2(h)), train=self.train, ratio=0.5)
        return self.fc3(h)

class PooledRecipeCuisine(BaseRecipe):
    def __init__(self, train, recipe_size, ingredient, output_size, pooling_type='max', features=1024):
        self.output_size = output_size
        self.pooling_type = pooling_type
        self.features = features
        self.recipe_size = recipe_size
        ouf = ingredient.out_feature * (2 if pooling_type == 'both' else 1)
        super(PooledRecipeCuisine, self).__init__(
            train=train,
            recipe_size=recipe_size,
            ingredient=ingredient,
            fc1=L.Linear(ouf, self.features),
            fc2=L.Linear(self.features, self.features),
            fc3=L.Linear(self.features, self.output_size)
        )


    def __call__(self, x):
        h = self.ingredient(x)
        if self.pooling_type == 'max':
            h = F.max_pooling_2d(h, (self.recipe_size,1))
        elif self.pooling_type == 'avg':
            h = F.average_pooling_2d(h, (self.recipe_size, 1))
        elif self.pooling_type == 'both':
            h1 = F.max_pooling_2d(h, (self.recipe_size,1))
            h2 = F.average_pooling_2d(h, (self.recipe_size, 1))
            h = F.array.concat.concat((h1, h2),1)
        h = F.dropout(F.relu(self.fc1(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc2(h)), train=self.train, ratio=0.5)
        h = self.fc3(h)
        return h


class MultiTaskPool(BaseRecipe):
    def __init__(self, train, recipe_size, ingredient, output_size_a, output_size_b, pooling_type='max', features=1024):
        self.output_size_a = output_size_a
        self.output_size_b = output_size_b
        self.pooling_type = pooling_type
        self.features = features
        self.recipe_size = recipe_size
        ouf = ingredient.out_feature * (2 if pooling_type == 'both' else 1)
        super(MultiTaskPool, self).__init__(
            train=train,
            recipe_size=recipe_size,
            ingredient=ingredient,
            fc1=L.Linear(ouf, self.features),
            fc_A_1=L.Linear(self.features, self.features),
            fc_A_2=L.Linear(self.features, self.output_size_a),
            fc_B_1 = L.Linear(self.features, self.features),
            fc_B_2 = L.Linear(self.features, self.output_size_b )
        )

        self.max = F.MaxPooling2D((recipe_size, 1))
        self.avg = F.AveragePooling2D((recipe_size, 1))

    def __call__(self, x):
        h = self.ingredient(x)
        if self.pooling_type == 'max':
            h = self.max(h)
        elif self.pooling_type == 'avg':
            h = self.avg(h)
        elif self.pooling_type == 'both':
            h1 = self.max(h)
            h2 = self.avg(h)
            h = F.array.concat.concat((h1, h2), 1)
        h = F.dropout(F.relu(self.fc1(h)), train=self.train, ratio=0.5)

        #path a
        a = F.dropout(F.relu(self.fc_A_1(h)), train=self.train, ratio=0.5)
        a = self.fc_A_2(a)

        b = F.dropout(F.relu(self.fc_B_1(h)), train=self.train, ratio=0.5)
        b = self.fc_B_2(b)


        return F.dstack((a,b))


class NaiveWider1DInceptionLayer(chainer.Chain):
    def __init__(self, in_channels, out1, out2, out3, out4, out5, out6, out7, conv_init=None, bias_init=None):
        depth = in_channels

        super(NaiveWider1DInceptionLayer, self).__init__(
            conv1=L.Convolution2D(depth, out1, (1, 1), initialW=conv_init, initial_bias=bias_init),
            conv2=L.Convolution2D(depth, out2, (1, 2), pad=(0, 0), initialW=conv_init, initial_bias=bias_init),
            conv3=L.Convolution2D(depth, out3, (1, 3), pad=(0, 1), initialW=conv_init, initial_bias=bias_init),
            conv4=L.Convolution2D(depth, out4, (1, 4), pad=(0, 1), initialW=conv_init, initial_bias=bias_init),
            conv5=L.Convolution2D(depth, out5, (1, 5), pad=(0, 2), initialW=conv_init, initial_bias=bias_init),
            conv6=L.Convolution2D(depth, out6, (1, 6), pad=(0, 2), initialW=conv_init, initial_bias=bias_init),
            conv7=L.Convolution2D(depth, out7, (1, 7), pad=(0, 3), initialW=conv_init,initial_bias=bias_init)
        )

    def __call__(self, c0):

        c1 = self.conv1(c0)

        c2 = self.conv2(c0)
        c2 = F.array.pad.pad(c2, ((0,0), (0,0), (0,0), (0,1)), 'constant')

        c3 = self.conv3(c0)

        c4 = self.conv4(c0)
        c4 = F.array.pad.pad(c4, ((0,0), (0,0), (0,0), (0,1)), 'constant')

        c5 = self.conv5(c0)

        c6 = self.conv6(c0)
        c6 = F.array.pad.pad(c6, ((0,0), (0,0), (0,0), (0,1)), 'constant')

        c7 = self.conv7(c0)

        return F.relu(F.array.concat.concat((c1, c2, c3, c4, c5, c6, c7),1))

class NaiveWide1DInceptionLayer(chainer.Chain):
    def __init__(self, in_channels, out1, out3, out5, out7, conv_init=None, bias_init=None):
        super(NaiveWide1DInceptionLayer, self).__init__(
            conv1=L.Convolution2D(in_channels, out1, (1, 1), initialW=conv_init, initial_bias=bias_init),
            conv3=L.Convolution2D(in_channels, out3, (1, 3), pad=(0, 1), initialW=conv_init, initial_bias=bias_init),
            conv5=L.Convolution2D(in_channels, out5, (1,5), pad=(0, 2), initialW=conv_init, initial_bias=bias_init),
            conv7=L.Convolution2D(in_channels, out7, (1, 7), pad=(0, 3), initialW=conv_init,initial_bias=bias_init)
        )

    def __call__(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        return F.relu(F.array.concat.concat((c1,c3,c5,c7),1))

class Inception1DLayer(chainer.Chain):
    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool, conv_init=None, bias_init=None):
        super(Inception1DLayer, self).__init__(
            conv1=L.Convolution2D(in_channels, out1, (1,1), initialW=conv_init, initial_bias=bias_init),
            proj3=L.Convolution2D(in_channels, proj3, (1,1), initialW=conv_init, initial_bias=bias_init),
            conv3=L.Convolution2D(proj3, out3, (1,3), pad=(0,1), initialW=conv_init, initial_bias=bias_init),
            proj5=L.Convolution2D(in_channels, proj5, (1,1), initialW=conv_init, initial_bias=bias_init),
            conv5=L.Convolution2D(proj5, out5, (1,5), pad=(0,2), initialW=conv_init, initial_bias=bias_init),
            projp=L.Convolution2D(in_channels, proj_pool, (1,1), initialW=conv_init, initial_bias=bias_init)
        )

    def __call__(self,x):
        out1 = self.conv1(x)
        out3 = self.conv3(F.relu(self.proj3(x)))
        out5 = self.conv5(F.relu(self.proj5(x)))
        pool = self.projp(F.max_pooling_2d(
            x, (1,3), stride=1, pad=(0,1)))
        y = F.relu(F.array.concat.concat((out1, out3, out5, pool), axis=1))

        return y

class IngredientCeption(BaseIngredientSentence):
    def __init__(self, train, alphabet_size, type='wide', feature_factor=1, dropout=False):
        self.feature_factor = feature_factor
        self.dropout = dropout
        if type == 'wider':
            l1 = NaiveWider1DInceptionLayer(alphabet_size, 16, 32, 48, 64, 64, 64, 48)
            input_2 = 336
        else:
            l1 = NaiveWide1DInceptionLayer(alphabet_size, 16, 128, 128, 64)
            input_2 = 336

        super(IngredientCeption, self).__init__(
            train=train,
            alphabet_size=alphabet_size,
            out_feature=256,
            inc1=l1,
            inc2a=Inception1DLayer(input_2, 32, 32, 64, 8, 16, 16),
            inc2b=Inception1DLayer(128, 64, 64, 96, 16, 48, 32),
            inc3=Inception1DLayer(240, 96, 48, 104, 8, 24, 32),
            inc4=Inception1DLayer(256, 96, 48, 104, 8, 24, 32),
        )

    def __call__(self, x):
        h = self.inc1(x)
        h = self.inc2a(h)
        h = self.inc2b(h)
        h = F.max_pooling_2d(h, (1,3))
        h = self.inc3(h)
        h = F.max_pooling_2d(h, (1,3))
        h = self.inc4(h)

        if self.dropout:
            h = F.dropout(h, train=self.train, ratio=0.5)
        h = F.max_pooling_2d(h, (1,15))

        return h


class SimpleIngredient(BaseIngredientSentence):
    def __init__(self, train, alphabet_size, layer_type='multiple', embed_input=False, depth='zero', width=128, inception=True, conv_init=None, bias_init=None):
        self.depth = depth # After input layer
        self.width = width
        self.embed_input = embed_input
        self.layer_type = layer_type
        kwargs = {}
        input = alphabet_size
        factor = 3 if layer_type == 'multiple-wide' else 1
        if self.embed_input:
            kwargs['embed'] = L.EmbedID(input+1,20,ignore_label=0)
            input = 20
        if self.depth in ['zero','one','two','three']:
            if inception:
                kwargs['l0'] = NaiveWider1DInceptionLayer(input, 16*factor, 32*factor, 48*factor, 64*factor, 64*factor, 64*factor, 48*factor)
            else:
                kwargs['l0'] = L.Convolution2D(input, 336 * factor, (1, 3), pad=(0, 1), initialW=conv_init,
                                               initial_bias=bias_init)
            input = 336 * factor

        if self.depth in ['one','two','three']:
            kwargs['l1'],input = self._get_layer(input, 1*factor)


        if self.depth in ['two','three']:
            self.width = self.width / 2
            kwargs['l2'],input = self._get_layer(input, 2*factor)
            kwargs['l3'],input = self._get_layer(input, 2*factor)

        if self.depth in ['three']:
            self.width = self.width / 2
            kwargs['l4'],input = self._get_layer(input, 4*factor)
            kwargs['l5'],input = self._get_layer(input, 4*factor)

        super(SimpleIngredient, self).__init__(
            train=train,
            alphabet_size=input,
            out_feature=input,
            **kwargs
        )

    def _get_layer(self, input, factor = 1):
        print self.layer_type
        if self.layer_type.startswith('multiple'):
            print "Adding multiple layer with factor %d to network" % factor
            return  NaiveWider1DInceptionLayer(input, 8 * factor, 16 * factor, 24 * factor, 32 * factor, 32 * factor, 32 * factor, 24 * factor), 168*factor
        else:

            print "Adding 1x3 layer with factor %d to network" % factor
            return L.Convolution2D(input, 256 * factor, (1, 3), pad=(0, 1)), 256*factor

    def __call__(self, x):
        if self.embed_input:
            x = self.embed(x)
            x = F.rollaxis(x,3,1)
        h = self.l0(x)

        if self.depth in ['one','two','three']:
            h = F.relu(self.l1(h))

        if self.depth in ['two','three']:
            h = F.max_pooling_2d(h, (1, 2), (1,2))
            h = F.relu(self.l2(h))
            h = F.relu(self.l3(h))


        if self.depth in ['three']:
            h = F.max_pooling_2d(h, (1, 2), (1, 2))
            h = F.relu(self.l4(h))
            h = F.relu(self.l5(h))

        return F.max_pooling_2d(h, (1, self.width))


class NutritionRegressionIngredient(BaseIngredientSentence):
    def __init__(self, alphabet_size, output_size, recipe_size, num_chars, depth='one', layer_type='multiple-filters',train=False):

        self.output_size = output_size
        self.recipe_size = recipe_size
        self.num_chars = num_chars
        self.width = num_chars
        self.layer_type = layer_type
        self.depth = depth
        kwargs = {}
        input = alphabet_size
        if self.depth in ['zero','one','two','three']:
            kwargs['l0'] = NaiveWider1DInceptionLayer(input, 16, 32, 48, 64, 64, 64, 48)
            input = 336

        if self.depth in ['one','two','three']:
            kwargs['l1'], input = self._get_layer(input, 1)


        if self.depth in ['two','three']:
            self.width = self.width / 2
            kwargs['l2'],input = self._get_layer(input, 2)
            kwargs['l3'],input  =self._get_layer(input, 2)

        if self.depth in ['three']:
            self.width = self.width / 2
            kwargs['l4'], input = self._get_layer(input, 4)

            kwargs['l5'],input = self._get_layer(input, 4)

        kwargs['conv_full_0'] = L.Convolution2D(input,input,(1,1))
        kwargs['conv_full_1'] = L.Convolution2D(input,1,(1,1))
        super(NutritionRegressionIngredient, self).__init__(
            train=train,
            alphabet_size=alphabet_size,
            out_feature=1,
            **kwargs
        )

    def _get_layer(self, input, factor = 1):
        print self.layer_type
        if self.layer_type.startswith('multiple'):
            print "Adding multiple layer with factor %d to network" % factor
            return  NaiveWider1DInceptionLayer(input, 8 * factor, 16 * factor, 24 * factor, 32 * factor, 32 * factor, 32 * factor, 24 * factor), 168*factor
        else:

            print "Adding 1x3 layer with factor %d to network" % factor
            return L.Convolution2D(input, 256 * factor, (1, 3), pad=(0, 1)), 256*factor

    def __call__(self, x):
        h = self.l0(x)
        
        if self.depth in ['one','two','three']:
            h = F.relu(self.l1(h))
            

        if self.depth in ['two','three']:
            h = F.max_pooling_2d(h, (1, 2), (1,2))
            h = F.relu(self.l2(h))
            h = F.relu(self.l3(h))
            


        if self.depth in ['three']:
            h = F.max_pooling_2d(h, (1, 2), (1, 2))
            h = F.relu(self.l4(h))
            h = F.relu(self.l5(h))
            

    
        h = F.max_pooling_2d(h, (1, self.width))
        h = self.conv_full_0(h)
        h = self.conv_full_1(h)
        h = F.reshape(h, h.shape[:-1])
        return h

class SummedRegressionRecipe(BaseRecipe):

    def __init__(self, train, recipe_size, ingredient, clip=True, **kwargs):
        super(SummedRegressionRecipe, self).__init__(train=train, recipe_size=recipe_size, ingredient=ingredient, **kwargs)
        self.clip = clip

    def __call__(self, x):
        h = self.ingredient(x)
        
        #if self.clip:
           # h = F.clip(h , 0.0, 1000000.0)

        x = F.sum(h, axis=2)

        return x

class RMSE(chainer.Chain):

    def __init__(self, predictor,output_size):
        super(RMSE, self).__init__(predictor=predictor)
        self.output_size = output_size

    def __call__(self, *args):
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        if self.output_size == 1:
            self.y = F.flatten(self.y)
            t = F.flatten(t)
        self.loss = F.sqrt(F.mean_squared_error(self.y, t))
        reporter.report({'loss': self.loss}, self)
        return self.loss


def get_nutrition_model(args, alphabet_size):
    from commoncrawl_dataset import get_fields
    fields = get_fields(args.dataset)
    network = args.network
    parts = network.split("-")
    types = {
        'multiple':'multiple',
        'multiple_wide': 'multiple-wide'
    }
    layer_type = types[parts[-2]] if parts[-2] in types else '3x3'

    if args.categories is None:
        ingredient = NutritionRegressionIngredient(alphabet_size, len(fields), args.num_ingredients, args.num_chars, parts[-1], layer_type)
        recipe = SummedRegressionRecipe(True,recipe_size=args.num_ingredients, ingredient=ingredient)
        model = RMSE(recipe, len(fields))
    else:
        ing = SimpleIngredient(True, alphabet_size, embed_input=False, depth=parts[-1], width=args.num_chars, inception=True, layer_type=layer_type)
        recipe = PooledRecipeCuisine(True, args.num_ingredients, ing, args.categories, 'both', 1024)
        model = L.Classifier(recipe)
    return model

def build_model(alphabet_size, output_size, recipe_size=32, recipe_type='fc', ing_type='wide', num_chars=128, embed=False, multi_task=False, load_file=None):

    if ing_type == 'wide':
        ing = IngredientSentenceSixLayers(True, alphabet_size)
    elif ing_type == 'shallow':
        ing = IngredientSentenceSmallerLayers(True, alphabet_size)
    elif ing_type == 'inception-wide-input':
        ing = IngredientCeption(True, alphabet_size, type='wider')
    elif ing_type == 'inception-wide-input-dropout':
        ing = IngredientCeption(True, alphabet_size, type='wider', dropout=True)
    elif ing_type == 'inception':
        ing = IngredientCeption(True, alphabet_size)
    elif ing_type[:6] == 'simple':
        pars = ing_type.split("-")
        inception = len(pars) < 3 or pars[2] == 'inc'
        ing = SimpleIngredient(True, alphabet_size,embed_input=embed, depth=pars[1],width=num_chars, inception=inception)

    else:
        raise Exception("ingredient type: %s not supported " % ing_type)

    types = recipe_type.split("-")
    type = 'max'
    features = 1024
    if len(types) > 0:
        type = types[0]
    if len(types) > 1:
        features = int(types[1])

    if type in ['max','avg','both']:
        if multi_task:
            recipe = MultiTaskPool(True, recipe_size, ing, output_size, output_size, type, features)
        else:
            recipe = PooledRecipeCuisine(True, recipe_size, ing, output_size, type, features)
    elif recipe_type == 'fc':
        recipe = FCRecipeCuisine(True, recipe_size, ing, output_size)
    else:
        raise Exception("recipe type: %s not supported " % recipe_type)
    if load_file is not None:
        load_npz(load_file, recipe)
    return recipe

from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class Classifier(link.Chain):

    """A simple classifier model.
    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.
    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.
    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        Returns:
            ~chainer.Variable: Loss value.
        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        print len(self.y.shape)
        print len(t.shape)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

if __name__ == '__main__':
    X = RMSE(F.square,1)
    x = np.array([4,2],dtype=np.float32)
    t = np.array([18,2],dtype=np.float32)
