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
        self.delay = (15, 45)
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
        if not kwargs['target']==self.bot.config['nick']:
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
            message = ' '.join(response)
            self.call_with_human_delay(self.add_to_chat, target, message, self.current)

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
            self.bot.privmsg(target, message)
            self.current = responding_to
            self.last_message = time.time()
        else:
            print "was responding to ",responding_to," with ",message," but that is no longer current"

