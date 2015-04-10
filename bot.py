# -*- coding: utf-8 -*-
import random
import time

from irc3.plugins.command import command
import irc3

import language

@irc3.plugin
class Plugin(object):

    def __init__(self, bot):
        #pprint.pprint(bot.config['corpus_file'])
        self.bot = bot
        grammar = language.CorpusAnalyzer(dirname=bot.config['corpus_dir'])
        vocabulary = grammar
        self.c = language.Conversation(grammar=grammar, vocabulary=vocabulary)
        self.delay = (5, 20)
        self.current = None
        self.last_message = time.time()
        self.break_silence_time = 20

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel):
        """Say hi when someone join a channel"""
        print "DEBUG: ", channel
        if mask.nick != self.bot.nick:
            self.bot.privmsg(channel, 'Hi %s!' % mask.nick)
        else:
            self.bot.privmsg(channel, 'Hi!')

    @irc3.event(r'^:(?P<mask>\S+!\S+@\S+) (?P<event>(PRIVMSG|NOTICE)) (?P<target>\S+) :\s*(?P<data>\S+.*)$')
    def msg3(self, **kwargs):
        # Only respond if this is one of the channels the bot has joined
        if kwargs['target'][1:].upper() in map(unicode.upper, self.bot.config['autojoins']):
            # don't respond to "Hi whoever!" messages
            if kwargs['data'].startswith("Hi ") and kwargs['data'].endswith("!"):
                print "Ignoring: ",kwargs['data']
                return 
            # don't respond to certain users (for example, the narrator)
            if kwargs['mask'].startswith("Narrator"):
                print "Ignoring the narrator"
                return
            # Listen to what has been said
            self.c.listen(kwargs['data'])
            toks = language.tokenize(kwargs['data'])
            self.current = toks
            self.last_message = time.time()
            self.generate_response(kwargs['target'])
            

    def generate_response(self, target):
        if not self.current:
            return
        response = self.c.three_words(first_word=self.current[-1], allow_longer=True)
        if response:
            self.call_with_human_delay(self.add_to_chat, target, response, self.current)

    @irc3.event(irc3.rfc.PING)
    def think(self, **kwargs):
        if time.time()-self.last_message>self.break_silence_time:
            if not self.current:
                self.current = ['the']
            self.generate_response('#%s'%self.bot.config['autojoins'][0])

    @irc3.extend
    def call_with_human_delay(self, func, *args, **kwargs):
        delay = random.randint(*self.delay)
        self.bot.loop.call_later(delay, func, *args, **kwargs)

    def add_to_chat(self, target, message, responding_to):
        if self.current == responding_to:
            message_str =' '.join(message)
            self.c.listen(message_str)
            self.bot.privmsg(target, message_str)
            self.current = responding_to
            self.last_message = time.time()
        else:
            print "was responding to ",responding_to," with ",message," but that is no longer current"

