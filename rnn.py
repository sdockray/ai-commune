import theano, theano.tensor as T
import numpy as np
from theano_lstm import RNN, LSTM

import nltk
from nltk.util import bigrams, trigrams, ngrams

import os, sys, glob, re, string, random, unicodedata, itertools, pickle
from pprint import pprint

from rnnmodel import Model

def pad_into_matrix(rows, padding = 0, force_width=0):
    if len(rows) == 0:
    	return np.array([0, 0], dtype=np.int32)
    lengths = [i for i in map(len, rows)]
    width = force_width if force_width else max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
    	if len(row)>width:
    		row = row[:width]
    	mat[i, 0:len(row)] = row
    return mat, list(lengths)

# Splits a sentence into phrases (as defined by punctuation)
def split_sentence(sentence):
	return filter(None, re.split("[" + string.punctuation + "]+", sentence))

# ngrams a sentence
def ngramize(sentence, n=4):
	return ngrams(nltk.word_tokenize(sentence), n)

# Shuffle 2 numpy arrays in unison
def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)


### Utilities:
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]
    
    def __init__(self, index2word = None, from_file=None):
        self.word2index = {}
        self.index2word = []
        
        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0
        
        if index2word is not None:
          self.add_words(index2word)

        if from_file is not None:
        	self.load(from_file)
                
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
                       
    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            try:
              return " ".join([self.index2word[word] for word in line])
            except:
            	print "Could not find ",word, " in ",line
            	return ""
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    try:
                      return " ".join([self.index2word[word] for word in line])
                    except:
                    	print "Could not find ",word
            indices = np.zeros(len(line), dtype=np.int32)
        elif type(line) is tuple:
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            #line = line.split(" ")
            line = nltk.word_tokenize(line)
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)

        return indices
    
    def save(self, save_loc):
    	path = "%s_vocab%s" % (os.path.splitext(save_loc)[0], os.path.splitext(save_loc)[1])
    	with open(path, 'wb') as f:
    		pickle.dump(self.index2word, f)
    	print "Saved vocabulary to ",path

    def load(self, load_loc):
			path = "%s_vocab%s" % (os.path.splitext(load_loc)[0], os.path.splitext(load_loc)[1])
			try:
				with open(path, 'rb') as f:
					print "Loading vocabulary from ",path
					self.index2word = pickle.load(f)
					for idx, word in enumerate(self.index2word):
						self.word2index[word] = idx
					print "... loaded %s words" % len(self.index2word)
			except:
				print "Tried to load vocabulary from %s but failed" % path


    @property
    def size(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.index2word)


class LanguageNN:

	def __init__(self, save_name='file.pkl', save_dir=None, corpus="", train_epochs=150, minibatch_size=8, vocab_size=25000):
		print "Starting..."
		self.MAX_VOCAB_SIZE = vocab_size
		self.TRAIN_EPOCHS = train_epochs
		self.MINIBATCH_SIZE = minibatch_size
		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			self.SAVE_NAME = os.path.join(save_dir, save_name)
		else:
			self.SAVE_NAME = save_name
		self.MAX_MEMORY = 30000 # sentences to keep in memory
		self.vocab = Vocab(from_file=self.SAVE_NAME)
		self.numerical_sequences = []
		self.numerical_sequences_matrix = None
		self.numerical_sequences_lengths = []
		
		# Ingest initial corpus
		contents = unicodedata.normalize('NFKD', corpus.decode("utf-8")).encode('ascii', 'ignore')
		sentences = nltk.sent_tokenize(contents)
		self.ingest_sentences(sentences)

		# construct model & theano functions:
		self.model = Model(
			input_size=10,
			hidden_size=10,
			vocab_size=self.MAX_VOCAB_SIZE,
			stack_size=1, # make this bigger, but makes compilation slow
			celltype=RNN # use RNN or LSTM
		)
		self.model.stop_on(self.vocab.word2index["."])
		self.model.load(self.SAVE_NAME)
		# train the model
		self.train_model(max_epochs=self.TRAIN_EPOCHS, samples_per_epoch=self.MINIBATCH_SIZE, reset_model=True)


	def train_model(self, max_epochs=100, samples_per_epoch=4, reset_model=False):
		if reset_model:
			max_epochs = max_epochs + self.model.epochs
		num_to_do = max_epochs - self.model.epochs
		if num_to_do<=0:
			print "No training to do: %s epochs and model is at %s"%(max_epochs, self.model.epochs)
			return
		starting_at = self.model.epochs
		try:
			for i in range(self.model.epochs, max_epochs):
				shuffle_in_unison(self.numerical_sequences_matrix, self.numerical_sequences_lengths)
				minibatch = self.numerical_sequences_matrix[:samples_per_epoch,:]
				self.model.update_fun(minibatch, self.numerical_sequences_lengths[:samples_per_epoch])
				self.model.epochs = i
				if (i - starting_at) % (max(1,num_to_do/10)) == 0:
					print i,"out of",max_epochs
					print "Example continuation from 'the':", self.continue_from("the")
		except KeyboardInterrupt:
			print "Model training interupted"
		print "Trained model for %s epochs in batches of %s" % (max_epochs, samples_per_epoch)
		# Save the model as it currently is
		self.model.save(self.SAVE_NAME, clean=True)
		# Probably should save the vocabulary as well!
		self.vocab.save(self.SAVE_NAME)


	def ingest_sentences(self, sentences):
		for s in sentences:
			self.vocab.add_words(nltk.word_tokenize(s))
		print "Vocabulary size: %s out of %s" % (len(self.vocab),self.MAX_VOCAB_SIZE)
		for s in sentences:
			self.numerical_sequences.append(self.vocab(s))
		self.numerical_sequences = self.numerical_sequences[:self.MAX_MEMORY]
		self.numerical_sequences_matrix, self.numerical_sequences_lengths = pad_into_matrix(self.numerical_sequences)
		pprint(self.numerical_sequences_matrix)


	def continue_from(self, starting_with, max_length=8, include_first_word=True):
		#print starting_with
		if type(starting_with) is list:
			tokens = starting_with
		else:
			tokens = nltk.word_tokenize(starting_with)
		sequence, s = self.random_sequence(self.vocab(tokens), choices=3, max_length=max_length+len(tokens)-1)
		if include_first_word:
			return self.vocab(sequence[len(tokens)-1:])
		else:
			return self.vocab(sequence[len(tokens):])

	def possible_sentences(self, starting_with):
		tokens = nltk.word_tokenize(starting_with)
		return self.vocab(self.random_sequence(self.vocab(tokens), choices=3, max_length=8))


	def random_sequence(self, starting_tokens, choices=4, max_length=20, score=0):
		if len(starting_tokens)>=max_length:
			return starting_tokens, score
		if starting_tokens[-1]==self.vocab.word2index["."]:
			return starting_tokens, score
		cs, s = self.numeric_continuations(starting_tokens, choices)
		score += s
		idx = random.randint(0,len(cs)-1)
		return self.random_sequence(np.append(starting_tokens,cs[idx]), choices, max_length, s)


	def numeric_continuations(self, numeric_tokens, num=4):
		# Gets items above a certain probability
		def filter_down(items,probabilities,cutoff=0.5):
			ret = []
			norm = [float(i)/max(probabilities) for i in probabilities]
			for i, n in zip(items, norm):
				if n>=cutoff:
					ret.append(i)
			return ret
		# Get continuations for a set of initial tokens
		numeric_tokens = np.append(numeric_tokens,self.vocab.word2index["."])
		#m, _ = pad_into_matrix(numeric_tokens)
		m = numeric_tokens
		predictions = self.model.pred_fun([m])
		"""
		import matplotlib.pyplot as plt
		data = np.random.random( (500,500) )
		arr = predictions[0,numeric_tokens.size-2]
		norm = [float(i)/max(arr) for i in arr]
		b = np.zeros((500, len(norm)))
		b[:,:] = norm
		plt.figimage(b)
		plt.savefig('zzz_image.png',format='png')
		"""
		try:
			arr = predictions[0,numeric_tokens.size-2]
			temp = np.argpartition(-arr, num)
			t2 = np.partition(-arr, num)
			return filter_down(temp[:num],-t2[:num],0.33), sum(-t2[:num])
		except:
			print "Oh no!"
			return []	


if __name__ == "__main__":
	theano.config.blas.ldflags="-lblas -lgfortran"
	corpus = sys.argv[1]
	dirname = 'corpuses/%s' % corpus
	txt_path = os.path.join(dirname,'*.txt')
	raw_text = ''
	print "Loading corpus matching: ",txt_path
	files = glob.glob(txt_path)   
	for name in files:
		try:
			with open(name) as f:
				raw_text = "%s\n%s" % (raw_text, f.read())
		except IOError as exc:
			if exc.errno != errno.EISDIR:
				raise # Propagate other kinds of IOError.
	# will create and train the model
	l = LanguageNN(corpus=raw_text, save_name="%s.pkl"%corpus, save_dir='nn', vocab_size=15000, train_epochs=1000, minibatch_size=4)
	l.continue_from("I am")


