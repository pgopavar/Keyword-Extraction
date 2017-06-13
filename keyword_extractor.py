import sys
import multiprocessing as mp
import random
import time
from time import sleep
from collections import defaultdict
import xml.etree.ElementTree as ET
from os import listdir
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer as Count
from sklearn.feature_extraction.text import HashingVectorizer as HV

from gensim import corpora
from gensim.models.ldamodel import LdaModel as LDA
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument as TD

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import *
from joblib import Parallel, delayed

random.seed(42086)

#Global Variables
st = PorterStemmer()
dictionary = None
w2v_model = None
max_files = -1
feature_count = None
model_type = None
n_keywords = None
n_folds = None
results_file = None
w2v_features = None
train_ids_w2v = None
test_ids_w2v = None
model_type_bkp = None

def edit_distance(s1, s2):
	m=len(s1)+1
	n=len(s2)+1

	tbl = {}
	for i in range(m): tbl[i,0]=i
	for j in range(n): tbl[0,j]=j
	for i in range(1, m):
		for j in range(1, n):
			cost = 0 if s1[i-1] == s2[j-1] else 1
			tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

	return tbl[i,j]


def timeit(method):

	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		print '%r %2.2f sec' % \
		      (method.__name__, te-ts)
		return result

	return timed


@timeit
def read_input_files():
	global max_files

	filelist = listdir('./alldata/')
	text = {}
	keywords = {}
	validfiles = []

	i = 0
	j = 0
	for fname in filelist:
		if 'PROQUEST' in fname:
			continue
		if i%10000 == 0:
			print i, j
		if max_files > 0:
			if j == max_files:
				break
		try:
			tree = ET.parse('./alldata/'+fname)
		except:
			i += 1
			continue
		root = tree.getroot()

		if root.find('Title') == None or root.find('Title').text == None:
			i += 1
			continue
		if root.find('Abstract') == None or root.find('Abstract').text == None:
			i += 1
			continue
		if root.find('Keyword') == None or root.find('Keyword').text == None:
			i += 1
			continue

		string = root.find('Title').text + '. ' + root.find('Abstract').text
		keys = []
		for k in root.findall('Keyword'):
			if k.text is None:
				continue
			keys.append(k.text.lower())
		text[j] = string
		keywords[j] = keys
		i += 1
		j += 1
		validfiles.append(i)

	keys = text.keys()

	return [text[k] for k in keys], [keywords[k] for k in keys]


def stem_doc(document):
	new_doc = ''
	for word in document.split():
		new_doc += st.stem(word) + ' '

	return new_doc


@timeit
def preprocess_text(documents, keywords):

	for i, key_set in enumerate(keywords):
		for j, key in enumerate(key_set):
			keywords[i][j] = key.strip().replace('-', ' ').replace('\n', ' ')

	key_freq = defaultdict(int)
	for i, key_set in enumerate(keywords):
		for j, key in enumerate(key_set):
			key_freq[key] += 1

	use_ids = set(range(len(documents)))

	#for i in range(len(documents)):
	#	new_set = []
	#	for k in keywords[i]:
	#		if key_freq[k] >= 2:
	#			new_set.append(k)
	#	keywords[i] = new_set
	#	if len(new_set) == 0:
	#		remove_ids.append(i)

	for i in range(len(documents)):
		for k in keywords[i]:
			if key_freq[k] < 3:
				use_ids.remove(i)
				break

	documents = [stem_doc(documents[i]) for i in use_ids]
	keywords = [keywords[i] for i in use_ids]

	return documents, keywords


@timeit
def construct_corpus(documents):
	global dictionary
	for i, doc in enumerate(documents):
		documents[i] = doc.split()
	dictionary = corpora.Dictionary(documents)
	corpus = [dictionary.doc2bow(doc) for doc in documents]

	return corpus


def get_tf_idf_features(documents):
	global feature_count
	global model_type_bkp

	if model_type_bkp == 0:
		Vectorizer = TFIDF(max_features = feature_count, stop_words = 'english', ngram_range=(1, 3))
	#Vectorizer = TFIDF(stop_words = 'english', ngram_range=(1, 2))
	else:
		Vectorizer = Count(stop_words = 'english', ngram_range = (1,2))
	#Vectorizer = HV(stop_words = 'english', ngram_range = (1,2))

	document_features = Vectorizer.fit_transform(documents)

	return document_features, [Vectorizer]


def get_lda_features(documents):
    global feature_count

    corpus = construct_corpus(documents)
    ldamodel = LDA(corpus, feature_count/10, iterations = 1000)
    lda_vectors = [ldamodel[doc_bow] for doc_bow in corpus]

    for i, lvec in enumerate(lda_vectors):
    	new_vec = [0] * (feature_count/10)
    	for (idx, val) in lvec:
    		new_vec[idx-1] = val
    	lda_vectors[i] = new_vec

    return lda_vectors, [ldamodel]


def doc2vec_features(documents):
    global feature_count

    tagged_documents = []
    for i, doc in enumerate(documents):
    	tdoc = TD(doc.split(), tags = [i])
    	tagged_documents.append(tdoc)

    doc2vec_model = Doc2Vec(tagged_documents, size = feature_count, iter = 5, workers = 4)
    doc2vec_features = [doc2vec_model.docvecs[i] for i in range(len(documents))]

    return doc2vec_features, [doc2vec_model]


def forloop(argtuple):
    global w2v_model
    global feature_count

    word_list = argtuple[0]
    doc_features = [0] * feature_count/10
    for j in range(len(word_list)):
    	try:
    		word_features = w2v_model[word_list[j]]
    	except:
    		continue
    	for k in range(feature_count/10):
    		doc_features[k] += word_features[k]
    doc_features = [doc_features[l]/float(len(word_list)) for l in range(len(doc_features))]
    return doc_features


@timeit
def word2vec_preprocessing(documents):
	global w2v_model
	global feature_count
	global w2v_features
	split_docs = [documents[i].split() for i in range(len(documents))]
	w2v_model = Word2Vec(split_docs, size = feature_count/10, workers = 32)
	print 'word2vec model built'
	pool = mp.Pool(4)
	w2v_features = pool.map(forloop, [(split_docs[i], i) for i in range(len(split_docs))])


@timeit
def wv_avg_features(documents):
	global w2v_model
	global w2v_features
	global train_ids_word2vec

	local_w2v_features = []
	for idx in train_ids_word2vec:
		local_w2v_features.append(w2v_features[idx])

	return local_w2v_features, [w2v_model]


@timeit
def transform_train_documents(documents):
	global model_type
	global w2v_model
	global w2v_features
	global train_ids_word2vec
	# Takes input strings, and returns documents as feature vectors
	# and the models list. 2 entries if tfidf + lda

	if model_type == 0:
		# Only TFIDF
		return get_tf_idf_features(documents)

	if model_type == 1:
		# Only LDA
		return get_lda_features(documents)

	if model_type == 3:
		# Only Doc2Vec
		return doc2vec_features(documents)

	if model_type == 2:
		tf_idf_features, tfidf_model_list = get_tf_idf_features(documents)
		lda_features, ldamodel_list = get_lda_features(documents)
		print tf_idf_features[0].__class__
		combined_features = [np.append(tf_idf_features[i].toarray()[0], lda_features[i]) for i in range(len(documents))]

		return (combined_features, [tfidf_model_list[0], ldamodel_list[0]])

	if model_type == 4:
		return wv_avg_features(documents)

	if model_type == 7:
		tfidf_feats, tfidf_model_list = get_tf_idf_features(documents)
		output_feats = [np.append(tfidf_feats[i].toarray()[0], w2v_features[train_ids_w2v[i]]) for i in range(len(train_ids_word2vec))]
		return output_feats, tfidf_model_list


@timeit
def transform_test_documents(documents, models):
	global dictionary
	global model_type
	global feature_count
	global test_ids_word2vec
	global w2v_features

	if model_type == 0:
		return models[0].transform(documents)

	if model_type == 1:
		all_test_vecs = []
		for document in documents:
			document_bow = dictionary.doc2bow(document.split())
			document_lda_vec = models[0][document_bow]
			output_vec = [0] * feature_count
			for (idx, val) in document_lda_vec:
				output_vec[idx] = val
			all_test_vecs.append(output_vec)
		return all_test_vecs

	if model_type == 2:
		all_test_vecs = []
		for document in documents:
			document_bow = dictionary.doc2bow(document.split())
	    	tfidf_part = models[0].transform([document]).todense()[0].tolist()[0]

	    	document_lda_vec = models[1][document_bow]
	    	lda_part = [0] * (feature_count/10)
	    	for (idx, val) in document_lda_vec:
	    		lda_part[idx] = val

	    	all_test_vecs.append(tfidf_part + lda_part)
		return all_test_vecs

	if model_type == 3:
		return [models[0].infer_vector(document.split()) for document in documents]

	if model_type == 4:
		test_features = []
		for idx in test_ids_word2vec:
			test_features.append(w2v_features[idx])

		return test_features

	if model_type == 7:
		test_features = []
		for i in range(len(test_ids_word2vec)):
			test_features.append(np.append(models[0].transform(documents[i]), w2v_features[test_ids_word2vec[i]]))
		return test_features

def find_similarity(document1, document2):
    return cosine_similarity(document1, document2)[0][0]
    #return 1/(manhattan_distances(document1, document2)[0][0])
    #return 1/(euclidean_distances(document1, document2)[0][0])


def train_full_model(documents, model_type = 3):
	pass


def evaluate_single_prediction(keywords_true, keywords_predicted, doc_text, sp_flag):
	global st
	tp = 0.0
	fp = 0.0
	total_p = len(keywords_predicted)

	for keyword in keywords_true:
		for key2 in keywords_predicted:
			dst = edit_distance(keyword, key2)

			if keyword == key2:
				tp += 1
				break

			elif dst < 3:
				tp += 1
				break

			elif st.stem(keyword) == st.stem(key2):
				tp += 1
				break

	if tp == 0:
		return 0, 0, 0, 0, 0

	precision = tp/len(keywords_predicted)
	recall = tp/len(keywords_true)
	fscore = (2 * precision * recall)/(precision + recall)

	mrr = 0.0
	for i, keyword in enumerate(keywords_predicted):
		if keyword in keywords_true:
			mrr += 1/(i + 1)
			break

	flags = []
	for keyword in keywords_predicted:
		if keyword in keywords_true:
			flags.append(1)
		else:
			flags.append(0)

	bpref = 0.0
	for i, flag in enumerate(flags):
		if flag == 1:
			bpref += 1 - ( (i - sum(flags[:i])) / (len(flags)))

	bpref = bpref/len(keywords_true)

	#for keyword in keywords_true[:1]:
	#	if keyword in keywords_predicted:
	#		mrr += 1/( keywords_predicted.index(keyword) + 1)
	#		print '-' + str(keywords_predicted.index(keyword))

	if (fscore > 0.7 and fscore < 0.9) or sp_flag:
		print doc_text
		print keywords_true
		print keywords_predicted
		print precision, recall, fscore, mrr, bpref
		if sp_flag:
			sys.exit()
		else:
			sleep(1)

	return precision, recall, fscore, mrr, bpref


@timeit
def evaluation(documents, keywords):
	global n_folds
	global model_type
	global n_keywords
	global train_ids_word2vec
	global test_ids_word2vec

	n_documents = len(documents)
	keys_set = set()
	ctr = 0
	for keys in keywords:
		for key in keys:
			keys_set.add(key)
			ctr += 1

	print '----------'
	print "Corpus Stats : "
	print n_documents, len(keys_set), ctr
	print '-----------'

	kf = KFold(n_documents, n_folds)

	results_data = {}
	fold_num = 0

	for (train_ids, test_ids) in kf:
		print 'FOLD STARTS'
		train_ids_word2vec = train_ids
		test_ids_word2vec = test_ids
		fold_num += 1
		results_data[fold_num] = {}

		train_docs_text = [documents[i] for i in train_ids]
		test_docs_text = [documents[i] for i in test_ids]
		train_keys = [keywords[i] for i in train_ids]
		test_keys = [keywords[i] for i in test_ids]

		train_doc_vectors, models = transform_train_documents(train_docs_text)
		test_doc_vectors = transform_test_documents(test_docs_text, models)

		similarities = cosine_similarity(test_doc_vectors, train_doc_vectors)

		try:
			train_size = train_doc_vectors.shape[0]
			test_size = test_doc_vectors.shape[0]
		except:
			train_size = len(train_doc_vectors)
			test_size = len(test_doc_vectors)

		keyword_scores = defaultdict(defaultdict)
		for i in range(test_size):
			keyword_scores[i] = defaultdict(list)
			for j in range(train_size):
				for keyword in train_keys[j]:
					keyword_scores[i][keyword].append(similarities[i][j])

			for key in keyword_scores[i].keys():
				if model_type_bkp == 5:
					keyword_scores[i][key] = sum(keyword_scores[i][key])
				else:
					keyword_scores[i][key] = sum(keyword_scores[i][key])



		for nkeys in [14]:
			fold_results = []
			for i in range(test_size):
				predicted_keywords = sorted(keyword_scores[i].keys(), key = lambda x : keyword_scores[i][x], reverse = True)[: nkeys]
				print i
				if i == 124:
					fold_results.append(evaluate_single_prediction(test_keys[i], predicted_keywords, test_docs_text[i], True))
				else:
					fold_results.append(evaluate_single_prediction(test_keys[i], predicted_keywords, test_docs_text[i], False))

			count = 0
			p_sum = 0.0
			r_sum = 0.0
			f_sum = 0.0
			mrr_sum = 0.0
			bpref_sum = 0.0
			for res_tuple in fold_results:
				p_sum += res_tuple[0]
				r_sum += res_tuple[1]
				f_sum += res_tuple[2]
				mrr_sum += res_tuple[3]
				bpref_sum += res_tuple[4]
				count += 1
			print p_sum/count, r_sum/count, f_sum/count, mrr_sum/count, bpref_sum/count

			results_data[fold_num][nkeys] = (p_sum/count, r_sum/count, f_sum/count, mrr_sum/count, bpref_sum/count)

	return results_data


def read_config_line(config_line):
	global max_files
	global feature_count
	global model_type
	global n_keywords
	global n_folds
	global results_file
	global model_type_bkp

	config_vals = config_line.split()

	max_files = int(config_vals[0])
	model_type = int(config_vals[1])
	feature_count = int(config_vals[2])
	n_keywords = int(config_vals[3])
	n_folds = int(config_vals[4])
	results_file = open('results.txt', 'a')

	if model_type == 5:
		model_type = 0
		model_type_bkp = 5
	if model_type == 6:
		model_type = 0
		model_type_bkp = 6
	else:
		model_type_bkp = model_type


def run(config_line):
	global results_file
	global model_type

	print config_line

	read_config_line(config_line)

	results_file.write(config_line)
	text_documents, keywords = read_input_files()
	text_documents, keywords = preprocess_text(text_documents, keywords)

	if model_type == 4 or model_type == 7:
		# Do all vectors calculation once
		word2vec_preprocessing(text_documents)


	results_data = evaluation(text_documents, keywords)

	results_by_nkey = {}
	for nkey in results_data[1].keys():
		p_sum = 0.0
		r_sum = 0.0
		f_sum = 0.0
		mrr_sum = 0.0
		bpref_sum = 0.0
		for fold_num in results_data.keys():
			p_sum += results_data[fold_num][nkey][0]
			r_sum += results_data[fold_num][nkey][1]
			f_sum += results_data[fold_num][nkey][2]
			mrr_sum += results_data[fold_num][nkey][3]
			bpref_sum += results_data[fold_num][nkey][4]
		results_by_nkey[nkey] = (p_sum/len(results_data.keys()), r_sum/len(results_data.keys()), f_sum/len(results_data.keys()), mrr_sum/len(results_data.keys()))

	print results_by_nkey
	results_file.write(str(results_by_nkey))
	results_file.write('\n')


if __name__ == '__main__':
	config_file = open('config.txt', 'r')
	for line in config_file:
		run(line)
