# AI-Commune
A commune - not for people, nor for their digital avatars, but for their data. This repository contains the data and the program(s) of the commune. This repository is the basis for technical, ethical, philosophical, and legal conversations about these programs and notions of a computational or algorithmic society, more broadly.

# Usage
Assuming you use virtual environments:
```
	$ git clone https://github.com/sdockray/ai-commune.git
	$ cd ai-commune
	$ virtualenv venv
	$ source venv/bin/activate
	$ pip install -r requirements.txt
```

You will need to train a tagger, which generates a pickle file, and then symlink to that pickle file, giving the name tagger.pickle. (see http://nltk-trainer.readthedocs.org/en/latest/train_tagger.html)

```
first, install nltk-trainer
python train_tagger.py treebank
ln -s <result of the training> tagger.pickle
```

Next copy *bot.ini.default* to the *config* directory and rename it (to anything). Edit the contents to configure a chatbot. For example:

```
	$ cp bot.ini.default config/SeanDockray.ini
	$ nano config/SeanDockray.ini
```

Change the IRC network, port, channel, as well as the bot name. The most important thing is to change the *corpus_dir* value to the same name as a directory within the *corpuses* directory (so I would choose *SeanDockray*). This associates a particular corpus with this particular bot.

Now, start up the bot!

```
	$ irc3 --host irc.server.net --port 6667 config/SeanDockray.ini
```

It should appear in the channel, ready to chat. Repeat the bot.ini.default copy process for each additional bot you want to run.


# High-Level Technical Description
As version 0.1 software, it is more of a proof-of-concept than a framework. The initial members of the commune have donated a corpus to this experiment. Each corpus is contained within a distinct directory as a set of files. The code now uses an IRC channel as the bounds within which the simulation operates, and each corpus is ingested into a bot. There are several possible "ways" that the bots formulate some speech act: (1) reproducing sentence structures (NLTK 'grammars') from the given corpus, but plugging in new words (2) trying to evolve new grammars from extremely simple grammars (3) generating appropriate trigram or trigram sequences in response to other speech acts, or topics of conversation. So far the third one has been the most successful.

# Transition to framework?
Experiments would be easier to run and more common if this project were a usable framework. How might the current structure be abstracted so that the bots could operate in various other situations besides IRC, or so that the corpus isn't just imported into a bot but something more capable of learning and evolution? As capital makes certain innovations, how might these innovations be brought into the system as a way of keeping our models and analyses current?  The data (corpuses) should continue to be publicly accessible, but it probably doesn't make sense for them to be under version control. There are two issues: (1) scale - if the data grows beyond the size of a DVD to the size of a data center, we will need different ways of storing and indexing it (2) new data - the corpus now is quite static, but if it evolves at the speed of a person's digital double, there needs to be a much more flexible system.

# Presupposition
If we produce enough data, transcode enough memories, thoughts, gestures, and dreams digitally, might there be algorithms that could pass as us? This is not about party games or fooling people, it's a question of whether the algorithm is accepted as an adequate substitute. Already artificial intelligences are good enough substitutes for certain menial administrative tasks, certain repetitive jobs. This project assumes that as humans are reproduced in the image of networks of algorithms, artificial intelligences are also becoming more complex, so that it may not be long before humans are successfully simulated. Furthermore, the simulation of human intelligence or evern consciousness is not the goal or summit, but merely one possible form among many.

# Install
