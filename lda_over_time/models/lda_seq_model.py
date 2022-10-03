"""
LdaSeqModel brings the Gensim's LdaSeqModel functionalities to our library.

Its main advantage over other models is that it can detect changes in the \
vocabulary used to describe each topic over time, making it more precise in \
classifying each topic. But it is slower to run.
"""
# IMPORTS
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldaseqmodel import LdaSeqModel as LdaSeqModel_
from lda_over_time.models.dtm_model_interface import DtmModelInterface
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import warnings


# TYPING
from typing import List, Optional


warnings.filterwarnings("ignore")


class LdaSeqModel(DtmModelInterface):
    """
    LdaSeqModel is a model that uses the Gensim's LdaSeqModel, which \
    supports the variance along time of the way that a certain topic is \
    approached (it can detect better the change of vocabulary to speak a \
    certain topic).

    With this feature, it may be more precise than PrevalenceModel, but \
    it is slower.

    :param corpus: List of documents' texts.
    :type corpus: list[str]

    :param dates: List of documents' publishing dates.
    :type dates: list[str]

    :param date_format: The date format used in `dates`, e.g. "%Y/%m/%d" for \
    "YYYY/MM/DD" format. More info at `documentation`_.
    :type date_format: str

    :param freq: The frequency used to group texts, e.g. "1M15D" for a \
    frequency of a month and 15 days. Useful notations: \
    day = "D"\
    month = "M"; \
    year = "Y". \
    More info at `pandas`_
    :type freq: str

    :param n_topics: Number of topics that the DTM model should find. The \
    default value is 100.
    :type n_topics: int, optional

    :param sep: Separator used to split each word, the default value is any \
    blank space.
    :type sep: str, optional

    :param workers: Number of workers (cpus) to use. If not provided, it will \
    use the total number of threads on running machine.
    :type workers: int, optional

    :return: Nothing
    :rtype: None

    .. _documentation: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    .. _pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """

    def __init__(self,
                 corpus: List[str],
                 dates: List[str],
                 date_format: str,
                 freq: str,
                 n_topics: int = 100,
                 sep: Optional[str] = None,
                 workers: Optional[int] = None):
        """
        I initialize the variables to train model.
        """
        # Save arguments inside object
        self.corpus = corpus
        self.dates = dates
        self.date_format = date_format
        self.freq = freq
        self.n_topics = n_topics
        self.sep = sep

        # Assign workers equals to arguments' if passed or set default
        self.workers = workers if workers is not None else cpu_count()


    def __prepare_corpus(self, corpus, dates, date_format, freq, sep):
        """
        This method prepares the corpus and dates in a supported format to
        LdaSeqModel.
        """
        # load dates with pandas with given format
        parsed_dates = pd.to_datetime(dates, format=date_format)

        # sort documents by date
        date_doc_df = pd.DataFrame({'date': parsed_dates,
                                    'text': corpus}).sort_values('date')

        # group by frequency
        grouped = date_doc_df.groupby(pd.Grouper(key='date', freq=freq),
                                      sort=True)

        # get length of each timeslice ordered by time
        timeslices = list(grouped.size().values)

        # get timestamps for all time slices
        timestamps = list(grouped.groups.keys())

        # Split words for each document
        data = [doc.split(sep) for doc in date_doc_df['text'].values]

        # Train Dictionary
        dictionary = Dictionary(data)

        # Convert corpus to list of BoWs
        corpus_bow = [dictionary.doc2bow(doc) for doc in data]

        # return values
        return corpus_bow, dictionary, timestamps, timeslices


    def __prepare_results(self,
                          model,
                          timestamps,
                          n_documents,
                          n_topics,
                          time_slices):
        """
        This method extracts the main topic for each document in corpus and \
        calculates the proportion of each topic in a specific time.
        """
        #
        # TODO: find better way of doing this (more performatic)
        #
        # get main topic for each document
        main_topics = []
        for i in range(n_documents):

            # get the distribution of topics over document
            distribution = model.doc_topics(i)

            # get main topic
            main = distribution.argmax()

            # save main topic
            main_topics.append(main)

        # calculate initial and final position to calculate for each time slice
        accumulated = np.add.accumulate([0] + time_slices)
        start_end = zip(accumulated[:-1], accumulated[1:])

        # list of proportions for each time slice
        proportions = []

        # calculate proportion for each timeslice
        for ini, end in start_end:

            # count occurrences for each topic
            count = np.bincount(main_topics[ini:end])

            # fill missing topics with zero
            count = np.append(count, [0] * (n_topics - len(count)))

            # calculate proportion of each topic in a specific time slice
            proportion = count / np.sum(count)

            # append a dictionary of dict[topic, proportion]
            proportions.append(dict(zip(range(n_topics), proportion)))

        # create dataframe
        result = pd.DataFrame.from_dict(proportions)

        # include field of timestamps
        result['date'] = timestamps

        # return dataframe of proportion of topics for each time slice
        return result

    def get_results(self):
        """
        This method should return a table representing the evolution of each \
        topic over time.

        :return: Returns a Pandas' dataframe where each column represents a \
        timeslice and must have a `date` and columns representing each \
        topics weight in that period.
        :rtype: pd.core.frame.DataFrame
        """
        return self.results


    def get_topic_words(self, topic_id, i, n):
        """
        This method returns the top n words that better describes the \
        topic in a specific time slice.

        :param topic_id: The id of the desired topic.
        :type topic_id: int

        :param i: The position of the desired timeslice in chronological \
        order the first (oldest) time slice is indexed by 1.
        :type i: int

        :param n: This specifies how many words that better describes the \
        topic at a specific time slice should be returned.
        :type n: int

        :return: It returns a list of top n words that best describes the \
        requested topic in a specific time.
        :rtype: list[str]

        """
        return [t[0] for t in self.model.print_topic(topic_id, i - 1, n)]


    @property
    def n_timeslices(self):
        """
        This attribute should be the number of timeslices found during \
        training.

        :return: It should return the number of time slices found in corpus. \
        :rtype: int
        """
        return len(self.time_slices)


    def prepare_args(self, i):
        """
        This method should return a dictionary with all necessary values to \
        call PyLdaVis.prepare method.

        :param i: The position of the desired timeslice in chronological \
        order, the first (oldest) time slice is indexed by 1.
        :type i: int

        :return: It returns a dictionary ready to be passed to PyLdaVis
        :rtype: dict[str, any]

        """
        # Calculate parameters
        doc_topics, topic_term, doc_lengths, term_frequency, vocab = \
                self.model.dtm_vis(time=i - 1, corpus=self.bow)

        # Return dictionary with parameters
        return {
            'topic_term_dists': topic_term,
            'doc_topic_dists': doc_topics,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency,
        }


    def train(self):
        """
        Train the DTM model.

        :return: Nothing.
        :rtype: None
        """
        # prepare data to use in training
        self.bow, self.dictionary, self.timestamps, self.time_slices = \
                self.__prepare_corpus(self.corpus,
                                      self.dates,
                                      self.date_format,
                                      self.freq,
                                      self.sep)

        # train dtm model
        self.model = LdaSeqModel_(corpus=self.bow,
                                  time_slice=self.time_slices,
                                  num_topics=self.n_topics,
                                  id2word=self.dictionary)

        # calculate result
        self.results = self.__prepare_results(self.model,
                                              self.timestamps,
                                              len(self.bow),
                                              self.n_topics,
                                              self.time_slices)

