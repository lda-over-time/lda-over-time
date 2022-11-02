"""
PrevalenceModel is a simpler and faster temporal LDA that returns the \
proportion of main topics in each time slice.

Its main advantage over other models is that it is fast. But it may not handle \
well the variation of the way that a topic is presented (when vocabulary to \
describe the topic varies over the given dataset).
"""
# IMPORTS
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from lda_over_time.models.dtm_model_interface import DtmModelInterface
from multiprocessing import cpu_count
from pyLDAvis.gensim_models import _extract_data as extract_data

import pandas as pd


# TYPING
from typing import List, Optional


class PrevalenceModel(DtmModelInterface):
    """
    PrevalenceModel is a simple temporal LDA model, it is faster, but \
    it may not handle well the evolution of topics (because the vocabulary \
    used in a certain topic may vary over time).

    :param corpus:Each item from the list is one document from corpus.
    :type corpus: list[str]

    :param dates: List of timestamps for each document in corpus, each date's \
    position should match with its respective text.
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

        # get number of parallel workers
        self.workers = workers if isinstance(workers, int) else cpu_count()


    def __normalize_lda_model(self, corpus, lda_model):
        """
        This method creates the dataframe from the result of lda training and \
        normalizes the weights (sum of weights for a topic = 1.0)

        :param corpus: List of Gensim's BoWs, where each BoW represents one \
        text from corpus.
        :type corpus: list[BoW]

        :param lda_model: Trained lda model
        :type lda_model: gensim.models.ldamulticore.LdaMulticore

        :return: Table of normalized results of Lda model.
        :rtype: pd.core.frame.DataFrame
        """
        # Create stream of weights for each document in corpus
        stream = lda_model.get_document_topics(corpus)

        # Get weights
        lda_weights = pd.DataFrame(map(dict, stream))

        # Normalize weights
        return (lda_weights.T / lda_weights.sum(axis=1)).T


    def __prepare_corpus(self, corpus, sep):
        """
        This method takes a list of document's string and convert them to \
        correct object typing asked by Gensim's LdaMulticore.

        :param corpus: List of documents' texts
        :type corpus: list[str]

        :param sep: Character that separates each word. If not given, all \
        whitespaces are considered separators.
        :type sep: str, optional

        :return: Pair of Gensim's Dictionary instance and list of list of \
        Gensim's BoW from corpus
        :rtype: tuple[gensim.corpora.Dictionary, list[BoW]]
        """
        # Split words for each document
        data = [doc.split(sep) for doc in corpus]

        # Train Dictionary
        dictionary = Dictionary(data)

        # Convert corpus to list of BoWs
        corpus_bow = [dictionary.doc2bow(doc) for doc in data]

        # Return dictionary and list of BoWs
        return dictionary, corpus_bow


    def __prepare_model_to_prevalence(self,
                                      dates,
                                      date_format,
                                      normalized_model,
                                      n_topics):
        """
        This method prepares the model to make a prevalence analysis over the \
        lda model. For this, it will create a table with two columns: one for \
        the date and other for the main topic of each document.

        :param dates: List of timestamps for each document in corpus, each \
        date's position should match with its respective text.
        :type dates: list[str]

        :param date_format:The date format used in `dates`.
        :type date_format: str

        :param normalized_model: Normalized result from lda model over corpus.
        :type: pd.core.frame.DataFrame

        :return: Table with date and main topic columns and each row \
        represents a document from corpus.
        :rtype: pd.core.frame.DataFrame
        """
        # load dates with pandas with given format
        dates = pd.to_datetime(dates, format=date_format)

        # get the main topic for each document
        main_topics = normalized_model[list(range(n_topics))].idxmax(axis=1)

        # join dates to main topics
        zipped = list(zip(dates.values, main_topics))

        # create table with date and topic
        return pd.DataFrame(zipped, columns=['date', 'main_topic'])


    def __prepare_pyldavis(self, dates, date_format, freq, ):
        """
        This method creates the list of periods found in corpus.

        :param dates: List of timestamps for each document in corpus, each date's \
        position should match with its respective text.
        :type dates: list[str]

        :param date_format:The date format used in `dates`.
        :type date_format: str

        :param freq:The frequency used to group texts.
        :type freq: str

        :return: translation of id to date and corpus grouped by timestamp
        """
        # TODO: improve this part...
        # load dates with pandas with given format
        timestamps = pd.to_datetime(dates, format=date_format)

        # create dataframe of dates
        timestamps_df = pd.DataFrame(timestamps, columns=['date'])

        # group dates by timeslice of lenght period
        dates_by_period = timestamps_df.groupby(
                pd.Grouper(key='date', freq=freq),
                sort=True
        )

        # get timestamps used to group
        timestamp_keys = sorted(dates_by_period.groups.keys())

        # initialize variables that hold translation of id to period and
        # group corpus by period
        id_to_period = {}
        corpus_by_period = {}

        # for each period: enumerate it and extract docs that belongs to this period
        for _id, key in enumerate(timestamp_keys):

            # enumerate pediod
            id_to_period[_id + 1] = key

            # get documents that belongs to this period
            indexes = dates_by_period.get_group(key).index

            # track documents of this period
            corpus_by_period[key] = list(map(
                lambda idx: self.bow[idx],
                indexes
            ))

        # return values
        return id_to_period, corpus_by_period


    def __train_lda_model(self, corpus, dictionary, n_topics):
        """
        This method trains the lda model.

        :param corpus: List of Gensim's BoWs, where each BoW represents one \
        text from corpus.
        :type corpus: list[BoW]

        :param dictionary: Dictionary with convertion id <-> word.
        :type dictionary: gensim.corpora.dictionary.Dictionary

        :param n_topics: Number of topics that the DTM model should find. \
        The default value is 100.
        :type n_topics: int, optional

        :return: It returns the trained Gensim's model LdaMulticore.
        :rtype: gensim.models.ldamulticore.LdaMulticore
        """
        return LdaMulticore(corpus=corpus,
                            id2word=dictionary,
                            num_topics=n_topics,
                            random_state=100,
                            chunksize=100,
                            passes=10,
                            minimum_probability=0.0,
                            per_word_topics=True,
                            workers=self.workers)


    def __train_prevalence(self,
                           docs_date_main_topic,
                           freq,
                           grouped_by_period):
        """
        This method takes the result of LDA and calculates the model of \
        temporal LDA with prevalence (frequency of main topics in a period).

        :param docs_date_main_topic: Table of documents with their date and \
        main topic
        :type docs_date_main_topic: pd.core.frame.DataFrame

        :param freq: Frequency to group documents
        :type freq: str

        :param grouped_by_period: Documents grouped by frequency
        :type grouped_by_period: pd.core.groupby.DataFrameGroupBy

        :return: It returns the temporal LDA trained model.
        :rtype: pd.core.frame.DataFrame
        """
        # Calculate number of posts per period
        posts_per_period = grouped_by_period.size().reset_index(name='count')

        # Group by period and topic
        grouped_by_period_topic = docs_date_main_topic.groupby(
            [
                pd.Grouper(key='date', freq=freq),
                pd.Grouper(key='main_topic')
            ],
            sort=True
        )

        # Calculate frequency of each main topic in a time interval period
        frequency = grouped_by_period_topic. \
                        size(). \
                        reset_index(name='count'). \
                        pivot(index='date',
                              columns='main_topic',
                              values='count').reset_index()

        # Replace nan with zero
        frequency = frequency.fillna(0.0)

        # Create table of prevalence
        prevalence = pd.DataFrame(
                (frequency[list(range(self.n_topics))].values.T / \
                                      posts_per_period['count'].values).T
        )

        # Add column with dates
        prevalence['date'] = frequency['date'].values

        # Return temporal lda model
        return prevalence


    @property
    def n_timeslices(self):
        """
        This attribute should be the number of timeslices found during \
        training.

        :return: It should return the number of time slices found in corpus. \
        :rtype: int
        """
        return self.grouped_by_frequency.ngroups


    def get_results(self):
        """
        This method should return a table representing the evolution of each \
        topic over time.

        :return: Returns a Pandas' dataframe where each column represents a \
        timeslice and must have a `date` and columns representing each \
        topics weight in that period.
        :rtype: pd.core.frame.DataFrame
        """
        return self.temporal_lda_model


    def get_topic_words(self, topic_id, i, n):
        """
        This method should return the top n words that better describes the \
        topic in a specific time slice.

        :param topic_id: The id of the desired topic.
        :type topic_id: int

        :param i: The position of the desired timeslice in chronological \
        order, the first (oldest) time slice is indexed by 1.
        :type i: int

        :param n:This specifies how many words that better describes the topic \
        at a specific time slice should be returned.
        :type n: int

        :return: It returns a list of top n words that best describes the \
        requested topic in a specific time.
        :rtype: list[str]
        """
        return [
                self.dictionary[word[0]]
                for word in self.lda_model.get_topic_terms(
                        topic_id,
                        n
                )
        ]


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
        # get timestamp
        period = self.id_to_timestamp[i]

        # return prepared data
        return extract_data(self.lda_model,
                            self.timestamp_to_corpus[period],
                            self.dictionary)


    def train(self):
        """
        This method trains the dtm model.

        :return: Nothing.
        :rtype: None
        """
        # Get dictionary and convert corpus to correct type
        self.dictionary, self.bow = self.__prepare_corpus(self.corpus,
                                                          self.sep)

        # train lda model
        self.lda_model = self.__train_lda_model(self.bow,
                                                self.dictionary,
                                                self.n_topics)

        # get normalized results of lda model from corpus
        self.normalized = self.__normalize_lda_model(self.bow,
                                                     self.lda_model)

        # prepare model to prevalence analysis
        self.docs_date_main_topic = \
                self.__prepare_model_to_prevalence(self.dates,
                                                   self.date_format,
                                                   self.normalized,
                                                   self.n_topics)
        # group documents by date
        self.grouped_by_frequency = \
                self.docs_date_main_topic.groupby(pd.Grouper(key='date',
                                                             freq=self.freq),
                                                  sort=True)

        # calculate temporal lda model
        self.temporal_lda_model = \
                self.__train_prevalence(self.docs_date_main_topic,
                                        self.freq,
                                        self.grouped_by_frequency)

        # get the list of timestamps
        self.id_to_timestamp, self.timestamp_to_corpus = \
                self.__prepare_pyldavis(self.dates, self.date_format, self.freq)

