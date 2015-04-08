import nltk
import itertools
import random
import unicodedata
import re
import copy
import string
import pprint
import os
import sys
import glob
import errno
#from nltk import parse_cfg, ChartParser
from random import choice

from nltk.grammar import CFG, Nonterminal, read_grammar, standard_nonterm_parser
from nltk.util import bigrams, trigrams
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from nltk.compat import (string_types, total_ordering, text_type, python_2_unicode_compatible, unicode_repr)
from nltk.corpus import wordnet

import rake
import pickle

tagger = pickle.load(open('tagger.pickle'))

def tokenize(raw_text, by_sentence=False, recursion=False):
	if raw_text:
		try:
			tokens = []
			if by_sentence:
				sents = nltk.sent_tokenize(raw_text)
				for sent in sents:
					tokens.append(nltk.word_tokenize(sent))
				return tokens
			else:
				return nltk.word_tokenize(raw_text)
		except:
			if recursion:
				return []
			else:
				contents = unicodedata.normalize('NFKD', raw_text.decode("utf-8")).encode('ascii', 'ignore')
				return tokenize(contents, by_sentence, recursion=True)

# tag a string, or an already tokenized text
def tag(text, by_sentence=True):
	tokens = tokenize(text, by_sentence)
	if by_sentence:
		tags = []
		for s in tokens:
			#tags.append(nltk.pos_tag(s))
			tags.append(tagger.tag(s))
			
	else:
		#tags = nltk.pos_tag(tokens)
		tags = tagger.tag(tokens)
	return tags

"""
	# Find all potential paths through a list of collocations pairs matching this grammar
	def find_sequences(self, colls_list):
		ret = []
		for colls,next_colls in zip(colls_list, colls_list[1:]+[]):
			sequences = []
			for coll in colls:
				for next_coll in next_colls:
					if coll[1].lower()==next_coll[0].lower():
						sequences.append([coll[0], coll[1], next_coll[1]])
			if not sequences:
				sequences.extend(colls)
			ret.append(sequences)
		return ret
"""

"""
A grammar is the basic unit for creating sentences and phrases. It is usually not created directly, but instead by
the Grammar Manager who feeds in the initial tokens (parts of speech representing a sentence) and a vocabulary corpus
from which the grammar can grab words. (Perhaps there should also be a fallback vocabulary, like brown?) 
Eventually, I will add the ability for the Grammar to mutate itself, adding in words or collapsing groups of words,
in order to modify the sentence structure and thus create new grammars.
"""
class Grammar(object):
	def __init__(self, tokens=[], vocabulary=None, *args, **kwargs):
		self.parts = tokens
		self.vocabulary = vocabulary

	# retrieves word collocations from a vocabulary given two POS, or 1 word and 1 POS
	# If it comes up empty, then it returns a random pair of words
	def get_collocations(self, pos_1=None, pos_2=None, word_1=None, word_2=None, how_many=20):
		candidates = None
		if pos_1 and pos_2:
			candidates = self.vocabulary.get_word_collocations(pos_1=pos_1, pos_2=pos_2, how_many=how_many)
		elif pos_1 and word_2:
			candidates = self.vocabulary.get_word_collocations(pos_1=pos_1, word_2=word_2, how_many=how_many)
		elif pos_2 and word_1:
			candidates = self.vocabulary.get_word_collocations(word_1=word_1, pos_2=pos_2, how_many=how_many)
		if not candidates:
			if pos_1 and not word_2:
				word_2 = self.vocabulary.get_random_word(pos_1)
			if pos_2 and not word_1:
				word_1 = self.vocabulary.get_random_word(pos_2)
			if not word_1:
				word_1 = '...'
			if not word_2:
				word_2 = '...'
			candidates = [(word_1, word_2)]
		#
		return candidates

	# Find all potential paths through a list of collocations pairs matching this grammar
	def find_sequences(self, colls_list):
		ret = []
		for colls,next_colls in zip(colls_list, colls_list[1:]+[]):
			sequences = []
			for coll in colls:
				for next_coll in next_colls:
					match = True
					for idx in range(1,len(coll)):
						word1 = coll[idx].lower()
						word2 = next_coll[idx-1].lower()
						if not word1==word2:
							match = False
					if match:
						new_sequence = coll[0:1] + next_coll[0:]
						sequences.append(new_sequence)

			if not sequences:
				sequences.extend(colls)
			ret.append(sequences)
		ret.append(next_colls)
		return ret

	# Randomly selects a sequence then moves forward to the next possible starting point
	# very basic - no attempt is made to connect the sequences
	def connect_sequences_basic(self, sequences):
		idx = 0
		splice_back = 0
		words = []
		last_position = None
		for position in sequences:
			if idx==len(words):
				# handle the very first position
				sequence = position[random.randint(0,len(position)-1)]
				words.extend(sequence)
			idx += 1
			last_position = position
		# finish up the sentence
		if len(words)<len(self.parts):
			sequence = last_position[random.randint(0,len(last_position)-1)]
			#print "Need to add ending to sentence: ", sequence
			words.extend(sequence[len(words)-len(self.parts):])
		return words

# Privileges the sequences coming after
	def connect_sequences_less_basic(self, sequences):
		idx = 0
		words = []
		last_position = None
		no_splice = False
		for position in sequences:
			if idx==0:
				# handle the very first position
				sequence = position[random.randint(0,len(position)-1)]
				words.extend(sequence)
			elif idx==len(words)-1:
				possibles = []
				for sequence in position:
					if sequence[0].lower()==words[-1]:
						possibles.append(sequence)
				if possibles:
					sequence = possibles[random.randint(0,len(possibles)-1)]
					if no_splice:
						words.extend(sequence)
					else:
						words[:-1].extend(sequence)
					no_splice = False
				elif no_splice:
					sequence = position[random.randint(0,len(position)-1)]
					words.extend(sequence)
					no_splice = False
				else:
					no_splice = True
				
			idx += 1
			last_position = position
		# finish up the sentence
		if len(words)<len(self.parts):
			sequence = last_position[random.randint(0,len(last_position)-1)]
			#print "Need to add ending to sentence: ", sequence
			words.extend(sequence[len(words)-len(self.parts):])
		if len(words)<len(self.parts):
			words.append(self.parts[-1])
		return words

	def connect_sequences(self, sequences, mode='basic'):
		if mode=='basic':
			return self.connect_sequences_basic(sequences)
		elif mode=='less_basic':
			return self.connect_sequences_basic(sequences)
		else:
			return self.connect_sequences_basic(sequences)

	def generate(self, keywords=None, num_samples=3):
		num_samples = 100
		if not self.parts or not self.vocabulary:
			return ''
		# Go through the sentence and look for suitable collocations
		words = []
		# Build a list of candidate pairs, which will be used to zip together a new sentence
		candidates = []
		prev_pos = None
		found_keywords = []
		for pos in self.parts:
			pairs = []
			if prev_pos:
				look_for_pairs = True
				if keywords:
					for k in keywords:
						a = self.vocabulary.get_word_collocations(word_1=k, pos_2=pos, how_many=num_samples)
						b = self.vocabulary.get_word_collocations(word_2=k, pos_1=prev_pos, how_many=num_samples)
						if a:
							pairs.extend( a )
						if b:
							pairs.extend( b )
						if a or b:
							if k not in found_keywords:
								look_for_pairs = False
							found_keywords.append(k)
				if look_for_pairs:
					pairs = self.get_collocations(pos_1=prev_pos, pos_2=pos, how_many=num_samples)
				candidates.append(pairs)
				prev_pos = pos
			else:
				prev_pos = pos
		# If there are keywords in the candidate pairs, we should weed out everything *else*
		#if keywords:

		# Get all 2 and 3 word sequences through this grammar
		sequences_3 = self.find_sequences(candidates)
		sequences_4 = self.find_sequences(sequences_3)
		sequences_5 = self.find_sequences(sequences_4)
		#pprint.pprint(sequences_5)		

		#pprint.pprint(sequences)
		
		sentence = self.connect_sequences(sequences_5, mode='less_basic')
		sentence = self.fix_sentence(sentence)
		return sentence
	
	# Does some massaging of a tokenized sentence to fix bad punctuation, etc.
	def fix_sentence(self, s):
		s = [w.replace('(', ',') for w in s]
		s = [w.replace(')', ',') for w in s]
		s = [w.replace(':', ',') for w in s]
		return s

	def __str__(self):
		return '|'.join([unicode_repr(x) for x in self.parts])

"""
A grammar manager holds a list of different grammars (each grammar makes a different sentence structure possible)
that it gets from a corpus. It is possible to give one corpus for grammar and another for the vocabulary, but by 
default they will be the same.
"""
class GrammarManager(object):
	def __init__(self, grammar=None, vocabulary=None, *args, **kwargs):
		self.grammars = []
		self.grammars_key = {}
		self.grammar_corpus = grammar
		if vocabulary:
			self.vocabulary_corpus = vocabulary
		else:
			self.vocabulary_corpus = self.grammar_corpus
		self.load_grammars()

	def load_grammars(self):
		if self.grammar_corpus and self.vocabulary_corpus:
			grammars = self.grammar_corpus.get_grammars()
			for g in grammars:
				grammar = Grammar(tokens=g, vocabulary=self.vocabulary_corpus)
				if str(grammar) not in self.grammars_key:
					self.grammars_key[str(grammar)] = grammar
					self.grammars.append(grammar)
				else:
					print "... ", str(grammar)," is already in the list of grammars - skipping"
			print "Loaded ",len(self.grammars)," different grammars"

	def random_sentence(self):
		g = self.grammars[random.randint(0, len(self.grammars)-1)]
		return g.generate()

	# This will return 3 words, or if a first word is specified it will return 3 words to follow the first word
	# If it can't find a continuation, then it will return None
	# Exclusions can be given to rule out repeats or anything else.
	# If allow_longer is set to true, then it might try and chain a couple more words on (randomly)
	def three_words(self, first_word=None, exclusions=[], allow_longer=False, topics=[], recursion=False):
		def check_cutoff_entities(phrase):
			lw = phrase[-1]
			for e in self.vocabulary_corpus.entities:
				start_appending = False
				for w in e:
					if start_appending:
						phrase.append(w)
					if w==lw:
						start_appending = True
				if start_appending:
					return phrase
			return phrase
		def finalize(phrase):
			if phrase[0] in ',?.!':
				phrase = phrase[1:]
			phrase = [w for w in phrase if w not in '()[]:;\'"' ]
			return check_cutoff_entities(phrase)

		possibles = self.vocabulary_corpus.get_trigram_collocations(word_1=first_word)
		filtered = [x for x in possibles if x not in exclusions]
		if filtered:
			phrase = filtered[random.randint(0,len(filtered)-1)]
			extra_words = 0
			if first_word:
				extra_words = 1
			if allow_longer:
				extra_words += random.randint(0,3)
			while extra_words>0:
				next_phrase = self.vocabulary_corpus.get_trigram_collocations(word_1=phrase[-2], word_2=phrase[-1])
				if next_phrase:
					phrase = phrase + (next_phrase[0][2],)
					extra_words -= 1
				else:
					extra_words = 0
			return finalize(phrase)
		else:
			# Try to do it based on synonyms of the first word
			if first_word and not recursion:
				syns = self.vocabulary_corpus.get_synonyms(first_word)
				for syn in syns:
					phrase = self.three_words(first_word=syn, exclusions=exclusions, allow_longer=allow_longer, recursion=True)
					if phrase:
						return finalize(phrase)
			# Try to do it based on the topics 
			if topics and not recursion:
				for topic in topics:
					syns = self.vocabulary_corpus.get_synonyms(first_word) + [topic,]
					for syn in syns:
						phrase = self.three_words(first_word=syn, exclusions=exclusions, allow_longer=allow_longer, recursion=True)
						if phrase:
							return finalize(phrase)
			return None
		
	# Will try to find a sentence that can reply suitably to given list of topics
	def topical_sentence(self, topics):
		g = self.grammars[random.randint(0, len(self.grammars)-1)]
		return g.generate(keywords=topics)

	# Rather than use the grammar, this uses trigrams from the vocabulary to construct a new sentence
	def markov_sentence(self, first_word=None, topics=[]):
		words = []
		ending = []
		done = False
		min_sentence_length = 7
		max_sentence_length = 150

		used = []
		def filter_used(trigrams):
			to_keep = []
			for t in trigrams:
				if not t[1:] in used:
					to_keep.append(t)
			return to_keep

		def filter_for_keywords(trigrams, keywords):
			if not keywords:
				return trigrams
			to_keep = []
			for t in trigrams:
				found = False
				for k in keywords:
					if not found:
						kl = k.lower()
						if t[0].lower()==kl or t[1].lower()==kl or t[2].lower()==kl:
							found = True
				if found:
					to_keep.append(t)
			if to_keep:
				return to_keep
			else:
				return trigrams

		# first construct an ending
		possibles = self.vocabulary_corpus.get_trigram_collocations(word_3='.')
		if possibles:
			possibles = filter_for_keywords(possibles, topics)
			last_part = possibles[random.randint(0, len(possibles)-1)]
			possibles = self.vocabulary_corpus.get_trigram_collocations(word_3=last_part[0])
			if possibles:
				possibles = filter_for_keywords(possibles, topics)
				prev_part = possibles[random.randint(0, len(possibles)-1)][:-1]
				used.append(prev_part)
				ending.extend(prev_part)
			used.append(last_part)
			ending.extend(last_part)
		
		while not done:
			if not words:
				if first_word:
					possibles = self.vocabulary_corpus.get_trigram_collocations(word_1=first_word)
				else:
					possibles = self.vocabulary_corpus.get_trigram_collocations()
			else:
				possibles = self.vocabulary_corpus.get_trigram_collocations(word_1=words[-1])
				if not possibles:
					possibles = self.vocabulary_corpus.get_trigram_collocations()
			if possibles:
				possibles = filter_used(possibles)
				possibles = filter_for_keywords(possibles, topics)
				if possibles:
					phrase = possibles[random.randint(0, len(possibles)-1)] if not words else possibles[random.randint(0, len(possibles)-1)][1:]
					used.append(phrase)
					words.extend(phrase)
			# look for an ending
			if len(words)>min_sentence_length:
				found = False
				for w in ending:
					if found:
						words.append(w)
					if words[-1].lower()==w.lower():
						found = True
				if found:
					done = True
			# force an ending
			if not done and len(words)>max_sentence_length:
				words.append('...')
				done = True
		return words

"""
A Conversation is a "subjective" model of a conversation (i.e. there is no master, objective model of a conversation)
which holds the recent terms, and idea of the overall topics, and the most recent things said by the one who has this 
model. The conversation will ask the grammar manager for new things to say. 
(Perhaps this could be extended to also include some AIML functionality if the conversation is a chat?)
"""
class Conversation(object):
	def __init__(self, grammar=None, vocabulary=None, memory_length=5, *args, **kwargs):
		self.memory = []
		self.topics = []
		self.gm = GrammarManager(vocabulary=vocabulary, grammar=grammar)
		self.memory_length = memory_length

	# This asks the grammar manager to use the various grammars to construct a new sentence
	def something_topical(self):
		return self.gm.topical_sentence(self.topics)

	# This asks the grammar manager to use trigrams in the vocabulary to construct a plausible-ish sentence
	def something_topical_2(self, first_word=None):
		return self.gm.markov_sentence(topics=self.topics, first_word=first_word)

	# This asks the grammar manager to use the various grammars to construct a new sentence
	def three_words(self, first_word=None, allow_longer=False):
		return self.gm.three_words(first_word=first_word, exclusions=[], topics=self.topics, allow_longer=allow_longer)

	# Takes a string
	def listen(self, s):
		self.memory.extend(tag(s))
		# [x[0] for x in tag(s, False)]
		# Keep the memory small-ish
		if len(self.memory)>self.memory_length:
			self.memory.pop(0)
		self.topics = []
		# add entities into the topic list
		self.topics.extend(self.extract_entities())
		# add keywords as well
		self.topics.extend(self.extract_keywords())

		# Extract entities (proper nouns)
	def extract_entities(self):
		entities = []
		for tagged in reversed(self.memory):
			chunks = nltk.ne_chunk(tagged)
			current_chunk = []
			for chunk in chunks:
				if isinstance(chunk[0], tuple):
					entities.extend([x[0] for x in chunk])
		return entities

	def extract_keywords(self):
		r = rake.RakeKeywordExtractor()
		entities = []
		for tagged in reversed(self.memory):
			for w in r.extract(' '.join([x[0] for x in tagged])):
				if w not in entities:
					entities.append(w)
		return entities

class EvolutionaryGrammar(CFG):
	def __init__(self, *args, **kwargs):
		super(EvolutionaryGrammar, self).__init__(*args, **kwargs)
		self.productions_k = {} #keyed productions
		self.corpus = None

	@classmethod
	def fromstring(self, input, encoding=None):
		start, productions = read_grammar(input, standard_nonterm_parser, encoding=encoding)
		cfg = EvolutionaryGrammar(start, productions)
		cfg.rebuild_keys()
		return cfg

	def tostring(self):
		result = ""
		for production in self._productions:
			if production.is_nonlexical():
				result += '\n    %s' % production
		return result

	# add in all the words from a corpus (for the POS specified by this grammar)
	def add_corpus(self, corpus_analyzer):
		self.corpus = corpus_analyzer
		self.learn_from_corpus()

	# by linking a corpus, the grammar will load terminals on demand
	def link_corpus(self, corpus_analyzer):
		self.corpus = corpus_analyzer
	
	# import one POS from corpus
	def corpus_import(self, pos, how_many=0):
		words = self.corpus.get_words(pos)
		imported = []
		for word in words:
			if how_many==0 or (how_many>0 and len(imported)<how_many):
				w = unicode(word.lower(), 'utf_8')
				if not self.production_exists(pos, w):
					self.add_production(pos, [w], rebuild=False)
					imported.append(w)
		print "Imported %s words (%s) from corpus: %s" % (len(imported), pos, ','.join(imported))
		self.productions_updated()

	# Learn N words from corpus for each POS in this grammar. By default, it will learn all of them
	def learn_from_corpus(self, N=0):
		keys = []
		for p in self.productions():
			if str(p.lhs()) not in keys:
				keys.append(str(p.lhs()))
			for t in p.rhs():
				if str(t) not in keys:
					keys.append(str(t))
		for k in keys:
			self.corpus_import(k, N)

	# Branches are names X0001... X9999
	def is_branch(self, t):
		keys = [str(x) for x in self.productions_k.keys()]
		if re.search(r"X\d{4}", str(t)):
			if str(t) in keys:
				return True
		return False

	# get a unique branch name
	def get_unique_branch_name(self):
		keys = [str(x) for x in self.productions_k.keys()]
		i = 1
		while "%s%s" % ("X", format(i, '04')) in keys:
			i += 1 
		return "%s%s" % ("X", format(i, '04'))

	# does a production exist in a grammar?
	def production_exists(self, lhs, rhs):
		new_p = "%s -> %s" % (lhs,rhs)
		for x in self._productions:
			if str(x)==new_p:
				return True
		return False

	# Add a new production. rhs must be a list
	def add_production(self, lhs, rhs, rebuild=True):
		if isinstance(lhs, basestring):
			lhs = nltk.grammar.Nonterminal(lhs)
		new_production = nltk.grammar.Production(lhs, rhs)
		self._productions.append(new_production)
		if rebuild:
			self.productions_updated()
		self.rebuild_keys()

	# Change the right hand side of a production. new_rhs must be a tuple
	def modify_production(self, production, new_rhs, rebuild=True):
		production._rhs = tuple(new_rhs)
		production._hash = hash((production._lhs, production._rhs))
		if rebuild:
			self.productions_updated()
		self.rebuild_keys()

	# Call this after the productions have been updated
	def productions_updated(self, calculate_leftcorners=True):
		self._categories = set(prod.lhs() for prod in self._productions)
		self._calculate_indexes()
		self._calculate_grammar_forms()
		if calculate_leftcorners:
			self._calculate_leftcorners()
		self.rebuild_keys()

	def rebuild_keys(self):
		self.productions_k = {}
		for p in self.productions():
			k = p.lhs()
			if not k in self.productions_k:
				self.productions_k[k] = []
			self.productions_k[k].append(p.rhs())
		#print self.productions_k

	def flat_structure(self):
		def iterate(nodes):
			flat = []
			no_branches = True
			for node in nodes:
				if self.is_branch(node):
					no_branches = False
					for p in self.productions(lhs=node):
						flat.extend(iterate(p.rhs()))
				else:
					flat.append(node)
			if no_branches and len(flat)==2:
				return [flat]
			else:
				return flat
		# Launch the recursive flattening
		return iterate(self.productions_k[self.start()][0])

	# Get a random sentence, but using the flattener & collocations
	def random_sentence_2(self, samples = 5):
		# Get a random word. A little logic to use collocations and the previous word to pick something viable
		def random_word(pos, curr_sentence=None):
			if curr_sentence:
				possibilities = self.corpus.get_word_collocations(first_word=curr_sentence[-1], pos_2=pos, how_many=samples)
				if len(possibilities)>0:
					return possibilities[random.randint(0,len(possibilities)-1)][1]
			return self.corpus.get_random_word(pos=pos)
		# Grab a collocation phrase from the parts of speech list given
		def make_phrase(pos_list, curr_sentence=None):
			ret = []
			if not len(pos_list)==2:
				return ret
			possibilities = self.corpus.get_word_collocations(pos_list[0], pos_list[1], how_many=samples)
			if len(possibilities)>0:
				return possibilities[random.randint(0,len(possibilities)-1)]
			else:
				return [random_word(pos_list[0], curr_sentence=curr_sentence), random_word(pos_list[1], curr_sentence=curr_sentence)]

		# I need a corpus
		if not self.corpus:
			return ""
		sentence = []
		current_group = []
		for word in self.flat_structure():
			if isinstance(word, Nonterminal):
				current_group.append(word)
				if len(current_group)==2:
					sentence.extend(make_phrase(current_group, curr_sentence=sentence))
					current_group = []
			else:
				if current_group:
					sentence.append(random_word(current_group[0], curr_sentence=sentence))
					current_group = []
				sentence.extend(make_phrase(word, curr_sentence=sentence))
		if current_group:
			sentence.append(random_word(current_group[0], curr_sentence=sentence))
		# Hope for the best!
		return ' '.join(sentence).lower()

	# generate a random sentence
	def random_sentence(self, depth=10):
		def do_one(item, depth):
			if depth > 0:
				if isinstance(item, Nonterminal):
					if item in self.productions_k:
						productions = self.productions_k[item]
						if self.is_branch(item):
							for prod in productions:
								for frag in do_all(prod, depth-1):
									yield frag
						else:
							prod = productions[random.randint(0,len(productions)-1)]
							for frag in do_all(prod, depth-1):
								yield frag
					else:
						yield []
				else:
					yield [item]

		def do_all(items, depth):
			if items:
				for frag1 in do_one(items[0], depth):
					for frag2 in do_all(items[1:], depth):
						yield frag1 + frag2
			else:
				yield []

		ret = ""
		for s in do_all(self.productions_k[self.start()][0], depth):
			ret += ' '.join(s)
		return ret

	def mutate(self):
		# add POS
		# add terminal
		# add nonterminal
		# modify nonterminal
		pass

	def mutate_fix_pos():
		pass

	# Splitting means to splice a POS in to a production, splitting the production into multiple
	def mutate_split_branch(self):
		# I need a corpus to mutate. Abort!
		if not self.corpus:
			return
		# simplify a branch to a single pos (is this on the left side of a tree? distill will always look right)
		def distill_branch(b, left=True):
			if not b in self.productions_k:
				return None
			looking_at = self.productions(lhs=b)[0].rhs()
			looking = True
			while looking:
				if left:
					looking_at = looking_at[len(looking_at)-1]
				else:
					looking_at = looking_at[0]
				if self.is_branch(looking_at):
					looking_at = self.productions(lhs=looking_at)[0].rhs()
				else:
					looking = False
			if not looking and looking_at:
				return looking_at
			return None
		# Performs a mutation
		def mutate(production):
			lhs = production.lhs()
			f1 = production.rhs()[0]
			f2 = production.rhs()[1]
			f1_pos = f1
			f2_pos = f2
			branch = nltk.grammar.Nonterminal(self.get_unique_branch_name())
			possibles = []
			if self.is_branch(f1):
				f1_pos = distill_branch(f1, left=True)
				possibles.extend(self.corpus.get_pos_collocations(pos_1=f1_pos, pos_3=f2_pos, how_many=3))
				possibles.extend(self.corpus.get_pos_collocations(pos_1=f1_pos, pos_2=f2_pos, how_many=3))
			elif self.is_branch(f2):
				f2_pos = distill_branch(f2, left=False)
				possibles.extend(self.corpus.get_pos_collocations(pos_2=f1_pos, pos_3=f2_pos, how_many=3))
			elif not self.is_branch(f1) and not self.is_branch(f2):
				possibles.extend(self.corpus.get_pos_collocations(pos_1=f1_pos, pos_3=f2_pos, how_many=3))
				possibles.extend(self.corpus.get_pos_collocations(pos_1=f1_pos, pos_2=f2_pos, how_many=3))
				possibles.extend(self.corpus.get_pos_collocations(pos_2=f1_pos, pos_3=f2_pos, how_many=3))				
			# Old way of calculating possibles
			print "Possible mutations of ",f1," and ",f2," into ", possibles
			#possibles = self.corpus.possible_pos_trigrams(f1_pos, f2_pos)				
			if possibles:
				random_possible = possibles[random.randint(0,len(possibles)-1)]
				print "Attempting to mutate ",f1," and ",f2," into ", random_possible
				# this random possible will be added to the 1st or the 2nd fragment
				# 1st: (POS, rhs[0], rhs[1]) or 2nd: (rhs[0], POS, rhs[1]) (rhs[0], rhs[1], POS)
				if str(f1_pos)==random_possible[1] and str(f2_pos)==random_possible[2]:
					pos = nltk.grammar.Nonterminal(random_possible[0])
					#print "a. modified production is %s -> %s %s" %(str(branch), str(pos), str(f1))
					#print "a. new production is %s -> %s %s" %(str(lhs), str(branch), str(f2))
					#self.add_production(branch, (pos,))
					self.add_production(branch, (pos, f1))
					self.corpus_import(pos, 3)
					self.modify_production(production, (branch, f2))
				else: # add it to the second fragment
					if str(f2_pos)==random_possible[2]:
						pos = nltk.grammar.Nonterminal(random_possible[1])
						#print "b. new production is %s -> %s %s" %(str(branch), str(pos), str(f2))
						#print "b. modified production is %s -> %s %s" %(str(lhs),str(f1), str(branch))
						#self.add_production(branch, (pos,))
						self.add_production(branch, (pos, f2))
						self.corpus_import(pos, 3)
						self.modify_production(production, (f1, branch))
					else:
						pos = nltk.grammar.Nonterminal(random_possible[2])
						#print "c. new production is %s -> %s %s" %(str(branch), str(f2), str(pos))
						#print "c. modified production is %s -> %s %s" %(str(lhs),str(f1), str(branch))
						#self.add_production(branch, (pos,))
						self.add_production(branch, (f2, pos))
						self.corpus_import(pos, 3)
						self.modify_production(production, (f1, branch))
		# Follow a non-terminal fragment (in other words, go down a branch)
		def follow(branch, max_depth=9):
			# We seem to be in a recursion, so let's bail
			if max_depth<=0:
				print "Possible recursion... giving up the chase"
				return
			productions = self.productions(lhs=branch)
			for p in productions:
				rhs = p.rhs()
				# we only know how to work with productions containing 2 fragments
				if len(rhs)==2:
					frag1_is_branch = self.is_branch(rhs[0])
					frag2_is_branch = self.is_branch(rhs[1])
					if not frag1_is_branch and not frag2_is_branch:
						# neither are branches, so mutate here
						return mutate(p)
					elif frag1_is_branch and frag2_is_branch:
						# pick one and go deeper
						return follow(rhs[random.randint(0,len(rhs)-1)], max_depth-1)
					else:
						# mutate here or follow the branch. which one?
						f = rhs[random.randint(0,len(rhs)-1)]
						if self.is_branch(f):
							return follow(f, max_depth-1)
						else:
							return mutate(p)
		# Main part of function
		result = follow(self.start())
		return result


# Base class for a grammar corpus
class GrammarCorpus(object):
	def get_grammars(self):
		return []

# Base class for a grammar corpus
class VocabularyCorpus(object):
	def get_random_word(self, pos):
		return ''
	def get_word_collocations(self, pos_1=None, pos_2=None, word_1=None, how_many=20):
		return []
	# get a list of synonyms
	def get_synonyms(self,word):
		syns = wordnet.synsets(word)
		return [l.name() for s in syns for l in s.lemmas()]


class CorpusAnalyzer(GrammarCorpus, VocabularyCorpus):
	def __init__(self, filename=None, dirname=None, *args, **kwargs):
		if dirname:
			raw_text = ''
			txt_path = os.path.join('corpuses',dirname,'*.txt')
			print "Loading corpus matching: ",txt_path
			files = glob.glob(txt_path)   
			for name in files:
				try:
					with open(name) as f:
						raw_text = "%s\n%s" % (raw_text, f.read())
				except IOError as exc:
					if exc.errno != errno.EISDIR:
						raise # Propagate other kinds of IOError.
		elif filename:
			raw_text = open(filename).read()
		# Initialize
		self.tags = []
		self.entities = []
		self.pos_map = {}
		self.word_bigrams = []
		self.pos_bigrams = []
		self.pos_trigrams = []
		self.pos_trigrams_cfd = None
		self.mixed_bigrams_cfd = None
		self.cfd = None # condition is POS, data is word
		self.word_cfd = None
		self.pos_cfd = None
		self.word_prob = None
		self.pos_prob = None
		self.collocations = None
		self.trigram_collocations = None
		self.tags = tag(raw_text)
		self.analyze()

	# Extract entities (proper nouns)
	def extract_entities(self):
		entities = []
		if self.tags:
			for sentence in self.tags:
				#chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
				chunks = nltk.ne_chunk(sentence)
				current_chunk = []
				for chunk in chunks:
					if isinstance(chunk[0], tuple):
						current_chunk.extend([x[0] for x in chunk])
					else:
						if current_chunk:
							entities.append(current_chunk)
						current_chunk = []
		self.entities = entities		

	def analyze(self):
		if self.tags:
			self.extract_entities()
			self.word_bigrams = itertools.chain(*[list(bigrams([x[0] for x in s])) for s in self.tags])
			#self.word_bigrams = bigrams([x[0] for x in self.tags])
			self.pos_bigrams = itertools.chain(*[list(bigrams([x[1] for x in s])) for s in self.tags])
			#self.pos_bigrams = bigrams([x[1] for x in self.tags])
			self.cfd = nltk.ConditionalFreqDist([(x[1], x[0]) for s in self.tags for x in s ])
			#self.cfd = nltk.ConditionalFreqDist([(x[1], x[0]) for x in self.tags])
			self.word_cfd = nltk.ConditionalFreqDist(self.word_bigrams)
			self.pos_cfd = nltk.ConditionalFreqDist(self.pos_bigrams)
			self.word_prob = nltk.ConditionalProbDist(self.word_cfd, nltk.MLEProbDist)
			self.pos_prob = nltk.ConditionalProbDist(self.pos_cfd, nltk.MLEProbDist)
			self.collocations = BigramCollocationFinder.from_words([(x[0].lower(),x[1]) for s in self.tags for x in s])
			self.trigram_collocations = TrigramCollocationFinder.from_words([x[0] for s in self.tags for x in s])
			self.pos_collocations = TrigramCollocationFinder.from_words([x[1] for s in self.tags for x in s])
			self.analyze_mixed_bigrams()
			self.analyze_pos_trigrams()

	# Returns all sentences as tokenized POS
	def get_grammars(self):
		if self.tags:
			return [[w[1] for w in s] for s in self.tags]
		return []
			
	# builds a lookup of what POS comes after particular words
	def analyze_mixed_bigrams(self):
		if self.tags:
			bigrams = []
			for s in self.tags:
				prev = None
				for x in s:
					if prev:
						bigrams.append((prev[0],x[1]))
					prev = x
			self.mixed_bigrams_cfd = nltk.ConditionalFreqDist(bigrams)
			#print self.filter_mixed_bigrams(pos='NN')
	
	# This returns pairs of words and parts of speech that immediately follow them
	def filter_mixed_bigrams(self, pos=None, max_items=3, min_freq=5):
		results = []
		for key in self.mixed_bigrams_cfd.conditions():
			if len(self.mixed_bigrams_cfd[key])<=max_items:
				for i in self.mixed_bigrams_cfd[key].items():
					if i[1]>=min_freq:
						if not pos:
							results.append((key, i[0]))
						elif str(pos)==i[0]:
							results.append((key, i[0]))
		return results

	# builds a lookup to find potential trigrams of POS
	def analyze_pos_trigrams(self):
		if self.tags:
			self.pos_trigrams = itertools.chain(*[list(trigrams([x[1] for x in s])) for s in self.tags])
			trigram_lookup = []
			for x in self.pos_trigrams:
				trigram_lookup.append(( "%s-%s"%(str(x[0]), str(x[1])), x ))
				trigram_lookup.append(( "%s-%s"%(str(x[0]), str(x[2])), x ))
				trigram_lookup.append(( "%s-%s"%(str(x[1]), str(x[2])), x ))
			self.pos_trigrams_cfd = nltk.ConditionalFreqDist(trigram_lookup)

	# Gets a bunch of common word collocations for 2 parts of speech
	def get_word_collocations(self, pos_1=None, pos_2=None, word_1=None, word_2=None, how_many=20):
		if self.collocations:
			bigram_measures = BigramAssocMeasures()
			c = copy.copy(self.collocations)
			if pos_1 and pos_2:
				c.apply_ngram_filter(lambda w1, w2: (w1[1],w2[1]) != (str(pos_1),str(pos_2)))
			elif word_1 and pos_2:
				c.apply_ngram_filter(lambda w1, w2: (w1[0],w2[1]) != (word_1,str(pos_2)))
			elif word_2 and pos_1:
				c.apply_ngram_filter(lambda w1, w2: (w1[1],w2[0]) != (str(pos_1),word_2))
			elif word_1:
				c.apply_ngram_filter(lambda w1, w2: (w1[0],0) != (word_1,0))
			results = c.nbest(bigram_measures.pmi, how_many)
			return [[w[0] for w in pair] for pair in results]
		else:
			return []

	# Gets a bunch of common word collocations for 2 parts of speech
	def get_pos_collocations(self, pos_1=None, pos_2=None, pos_3=None, how_many=20):
		if self.pos_collocations:
			trigram_measures = TrigramAssocMeasures()
			c = copy.copy(self.pos_collocations)
			if not pos_1:
				c.apply_ngram_filter(lambda w1, w2, w3: (0,w2,w3) != (0,str(pos_2),str(pos_3)))
			elif not pos_2:
				c.apply_ngram_filter(lambda w1, w2, w3: (w1,0,w3) != (str(pos_1),0,str(pos_3)))
			elif not pos_3:
				c.apply_ngram_filter(lambda w1, w2, w3: (w1,w2,0) != (str(pos_1),str(pos_2),0))
			else:
				c.apply_ngram_filter(lambda w1, w2, w3: (w1,w2,w3) != (str(pos_1),str(pos_2),str(pos_3)))
			results = c.nbest(trigram_measures.pmi, how_many)
			return results
		else:
			return []

	# Gets a bunch of common word collocations for 2 parts of speech
	def get_trigram_collocations(self, word_1=None, word_2=None, word_3=None, stop=False, how_many=20):
		if not word_1 and not word_2 and not word_3:
			starting_words = [s[0][0] for s in self.tags]
			word_1 = starting_words[random.randint(0, len(starting_words)-1)]
		if self.trigram_collocations:
			trigram_measures = TrigramAssocMeasures()
			c = copy.copy(self.trigram_collocations)
			if word_1 and word_2 is None and word_3 is None:
				c.apply_ngram_filter(lambda w1, w2, w3: (w1,None,None) != (word_1,None,None))
			if word_1 and word_2 and word_3 is None:
				c.apply_ngram_filter(lambda w1, w2, w3: (w1,w2,None) != (word_1,word_2,None))
			if word_1 is None and word_2 is None and word_3:
				c.apply_ngram_filter(lambda w1, w2, w3: (None,None,w3) != (None,None,word_3))
			if not word_3 and not stop:
				c.apply_ngram_filter(lambda w1, w2, w3: u'.' in [w1,w2,w3])
			results = c.nbest(trigram_measures.pmi, how_many)
			return results
		return []

	# for any 2 parts of speech (in order) what are the possible trigrams they are a part of?
	def possible_pos_trigrams(self, pos_1, pos_2):
		condition = "%s-%s" % (str(pos_1), str(pos_2))
		if condition in self.pos_trigrams_cfd.conditions():
			return self.pos_trigrams_cfd[condition].items()
		return []

	# POS bigram probabilities
	def pos_bigram_probabilities(self):
		probabilities = []
		for c in self.pos_cfd.conditions():
			for i in self.pos_cfd[c].keys():
				prob = self.pos_cfd[c].freq(i)
				prob = 0.0001 if not prob else prob
				probabilities.append(((c, i), prob))
		sorted_probabilities = sorted(probabilities, key = lambda pair:pair[1], reverse = True)
		return sorted_probabilities
	
	# get all words for a POS
	def get_words(self, pos, include_counts=False):
		if str(pos) in self.cfd.conditions():
			items = sorted(self.cfd[str(pos)].items(), key = lambda pair:pair[1], reverse = True)
			if include_counts:
				return items
			else:
				return [x[0] for x in items]
		else:
			return []

	# get a random word for a POS (could also do getting a probable word?)
	def get_random_word(self, pos):
		if str(pos) in self.cfd:
			words = self.cfd[str(pos)].keys()
			if words:
				return words[random.randint(0,len(words)-1)]
		return None

	def pos_after(self, condition):
		if condition in self.pos_cfd:
			return self.pos_cfd[condition].items()
		else:
			return []

	def word_after(self, condition):
		if condition in self.word_cfd:
			return self.word_cfd[condition].items()
		else:
			return []

	# probability that w2 follows w1
	def words_probability(self, w1, w2, pos=False):
		if not pos and w1 in self.word_prob:
			return self.word_prob[w1].prob(w2)
		elif pos and w1 in self.pos_prob:
			return self.pos_prob[w1].prob(w2)
		else:
			return 0

	# probability of a sentence is the average probability of all its word pairs
	def sentence_probability(self, sentence, pos=False):
		if pos:
			words = [x[1] for x in tag(sentence, False)]
		else:
			words = tokenize(sentence, False)
		pairs = bigrams(words)
		total = 0
		count = 0
		for pair in pairs:
			total += self.words_probability(pair[0], pair[1], pos)
			count += 1
		if count>0:
			return total/count
		else:
			return 0

	# A negative rating
	def calculate_penalties(self, sentence, penalty=0.2, threshhold=0.05):
		retval = 0
		words = [x[1] for x in tag(sentence, False)]
		pairs = bigrams(words)
		for pair in pairs:
			if self.words_probability(pair[0], pair[1], True)<threshhold:
				retval += penalty
		return retval

	# combines the two kinds of probability
	def score_sentence(self, sentence):
		return self.sentence_probability(sentence) + self.sentence_probability(sentence, True)

demo_grammar = """
  S -> X01 VBD
  X01 -> DT NN
  X02 -> IN X01
  VBD -> 'slept' | 'saw' X01 | 'walked' X02
  DT -> 'the' | 'a'
  NN -> 'man' | 'park' | 'dog'
  IN -> 'in' | 'with'
"""
demo_grammar_small = """
    S -> NN X0001
    X0001 -> X0002 X0003
    X0002 -> NNP NNP
    X0003 -> VBD VBN
"""

#grammar = EvolutionaryGrammar.fromstring(demo_grammar_small)
#print grammar.tostring()
#print grammar.flat_structure()
#print str(grammar)
#grammar.random_sentence()


"""
# add a new noun to the vocabulary
lhs = nltk.grammar.Nonterminal('NN')
rhs = [u'authorship']
new_production = nltk.grammar.Production(lhs, rhs)
grammar._productions.append(new_production)    
print str(grammar)


corpus = CorpusAnalyzer('bratton.txt')
c = Conversation(grammar=corpus)
c.listen("can you see a ghost in the network")
c.listen("we are machines of loving grace in the school of life")
c.listen("Bruce Sterling")
print c.something_topical_2(first_word="is")


import time

corpus_bratton = CorpusAnalyzer('bratton.txt')
corpus_dockray = CorpusAnalyzer('dockray.txt')
bratton = Conversation(grammar=corpus_bratton)
dockray = Conversation(grammar=corpus_dockray)
w1 = None
w2 = None

while True:
	if not w1:
		w1 = [None,None,None]
	w2 = bratton.three_words(first_word=w1[-1], allow_longer=True)
	print "bratton: ",w2
	time.sleep(2)
	if not w2:
		w2 = [None,None,None]
	w1 = dockray.three_words(first_word=w2[-1], allow_longer=True)
	print "dockray: ",w1
	time.sleep(2)
"""

#grammar.link_corpus(corpus)
#grammar.learn_from_corpus(20)


#idx = 81 #random.randint(0,len(corpus.tags)-1)
#print corpus.tags[idx]
#print corpus.get_word_collocations(pos_1='CC', pos_2='WP')
#print corpus.get_word_collocations(pos_1='WP', pos_2='VBZ')
#print corpus.get_word_collocations(pos_1='VBZ', pos_2='DT')
#print corpus.get_word_collocations(pos_1='DT', pos_2='NN')


"""
def print_random_sentence(N=10):
	tot = 0
	for i in range(N):
		s = grammar.random_sentence_2(samples=25)
		score = corpus.score_sentence(s)
		tot += score
		print s
	return tot/N

score = print_random_sentence(10)
print "Average score = ", score

grammar.mutate_split_branch()
print grammar.flat_structure()

score = print_random_sentence(10)
print "Average score = ", score

print grammar.tostring()
"""
#print corpus.pos_after('NN')
#print corpus.sentence_probability('the essay form is a cheese slice', pos=True)
#print corpus.sentence_probability('oodles of noodles and poodles with snickerdoodles', pos=True)
#print corpus.sentence_probability('walter benjamin wrote that eggs are easy to eat but hard to chew', pos=True)
#print corpus.get_words('MD')
#print corpus.word_after('could')
#print corpus.get_words('RBS')
#print corpus.get_random_word('JJ')


"""
f = file('test2.txt','r')
contents = f.read()
contents = unicodedata.normalize('NFKD', contents.decode("utf-8")).encode('ascii', 'ignore')
counts = get_tags(contents)
for tag in counts:
	print tag, len(counts[tag])
# get a random noun
print counts['NN'][random.randint(0,len(counts['NN'])-1)]
print does_production_exist(grammar, 'NN', "'park'")
"""

from nltk.stem.wordnet import WordNetLemmatizer


