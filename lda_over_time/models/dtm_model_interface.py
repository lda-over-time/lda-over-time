"""
DtmModelInterface is an interface to create new DTM modules, these modules
should train the model and return values to front end.
"""

from lda_over_time.errors.not_implemented import NotImplemented

import inspect


class DtmModelInterface:
    """
    DtmModelInterface defines methods and attributes that a module should have \
    in order to be passed to front end.

    :param corpus: Each item from the list is one document from corpus.
    :type corpus: list[str]

    :param dates: List of timestamps for each document in corpus, each \
    date's position should match with its respective text.
    :type dates: list[str]

    :param date_format: The date format used in `dates`.
    :type date_format: str

    :param freq: The frequency used to group texts.
    :type freq: str

    :param n_topics: Number of topics that the DTM model should find. The \
    default value is 100.
    :type n_topics: int, optional

    :param sep: Separator used to split each word, the default value is any \
    blank space.
    :type sep: str, optional

    :param workers: Number of workers (cpus) to use. If not provided, it \
    will use the value of multiprocessing.cpu_count()
    :type workers: int, optional

    """

    @property
    def n_timeslices(self):
        """
        This attribute should return the number of timeslices found.

        :return: It should return the number of time slices found in corpus.
        :rtype: int

        """
        # Get method's name
        method_name = inspect.currentframe().f_code.co_name

        # It was not overwritten by child: raise exception
        raise NotImplemented(method_name)


    def get_results(self):
        """
        This method should return a table representing the evolution of each \
        topic over time.

        :return: It must return a Pandas' dataframe where rows represents \
        different time slices and they are sorted by date, it must have one \
        column `data` and the remaining columns numbered from 0 to k (number \
        of topics - 1) that holds weights of each topic in that period.
        :rtype: pandas.core.frame.DataFrame

        """
        # Get method's name
        method_name = inspect.currentframe().f_code.co_name

        # It was not overwritten by child: raise exception
        raise NotImplemented(method_name)


    def get_topic_words(self, topic_id, i, n):
        """
        This method should return the top n words that better describes the \
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
        # Get method's name
        method_name = inspect.currentframe().f_code.co_name

        # It was not overwritten by child: raise exception
        raise NotImplemented(method_name)


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
        # Get method's name
        method_name = inspect.currentframe().f_code.co_name

        # It was not overwritten by child: raise exception
        raise NotImplemented(method_name)


    def train(self,
              corpus,
              dates,
              date_format,
              freq,
              n_topics=100,
              sep=None,
              workers=None):
        """
        This method should train the dtm model.

        :return: nothing
        :rtype: None

        """
        # Get method's name
        method_name = inspect.currentframe().f_code.co_name

        # It was not overwritten by child: raise exception
        raise NotImplemented(method_name)
