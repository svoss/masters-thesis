import os
import sys
import argparse
from os.path import dirname,realpath
path = os.path.join(dirname(dirname(realpath(__file__))), '../code')
sys.path.append(path)
from commoncrawl_dataset import get_nutrition_dataset
def build_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--latex',action='store_true')
    parser.add_argument('--num-chars',type=int, default=96)
    parser.add_argument('--num-ingredients',type=int, default=24)
    parser.add_argument('--categories',type=int,default=None)
    parser.add_argument('--dataset',type=str,default='calories')
    parser.add_argument('--test',action='store_true')

    parser.add_argument('--limit-to',type=int, default=None)
    return parser.parse_args()

def _to_str(cls):
	return "%.2f - %.2f" % (cls[0],cls[1])

def to_latex(classes):
	print " & ".join(["\\textbf{%s}" %  k for k in classes.keys() if k != 'recipe_yield']) + "\\\\"
	for i in range(len(classes[classes.keys()[0]])):
		print " & ".join([ _to_str(c[i]) for k,c in classes.iteritems() if k != 'recipe_yield']) +"\\\\"
			
def determine_recipe_yield_classes():
	from config import get_mongo_db
	db = get_mongo_db()
	query = {"context.language":"en","parsed.recipe_yield":{"$ne":None}, 'max_ingredient_size': {"$lte":args.num_chars},'recipe_size':{"$lte":args.num_ingredients},'recipe_size':{"$gt":0}}
	recipes = db['recipes'].find(query)
	yields = []
	for r in recipes:
		yields.append(int(r['parsed']['recipe_yield']))
	covered = 0
	class_coverage = 0
	per_step = len(yields) // 10
	this_step = 0
	next_step =  per_step
	classes = [0]
	current_class = 0
	custom_classes = [[0,1],[2,3],[4,4],[5,6],[7,8],[9,11],[12,14],[15,19],[20,29],[30,72]]
	for i in xrange(int(max(yields))):
		yield_coverage = len([y for y in yields if y == i])
		C = custom_classes[current_class]
		#print i,class_coverage, class_coverage/float(len(yields))
		covered += yield_coverage
		class_coverage += yield_coverage
		if i == C[1]:

			print "%d-%d" % tuple(C) , '&', "%.2f\\%%\\\\" % ((float(class_coverage)/len(yields)) * 100.0)
			current_class += 1
			next_step += per_step
			classes.append(i)
			this_step = 0
			class_coverage = 0

	print "%d-%d" % tuple(C) , '&', "%.2f %%" % ((float(class_coverage)/len(yields)) * 100.0)

if __name__ == "__main__":
	args = build_args()
	determine_recipe_yield_classes()


	#train,_,_,_ = get_nutrition_dataset(args)
	#classes = train.db.y_encoder.classes
	#if args.latex:
	#	to_latex(classes)
	#else:
	#	print classes
