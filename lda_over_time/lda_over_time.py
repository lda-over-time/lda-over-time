"""
LdaOverTime is a framework that brings an easier way of doing Topic Modeling \
Analysis Over Time and get visualization of results.

In brief, Topic Modeling is a technique that finds topics that each document \
from a collection covers. And, by addind the time in this equation, we can \
study how much and why one certain topic is more or less discussed in a \
time slice.
"""
# IMPORTS
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pyLDAvis
import seaborn as sns


# TYPING
from lda_over_time.models.dtm_model_interface import DtmModelInterface
from typing import List, Optional



# CODE
class LdaOverTime:
    """
    LdaOverTime provides an easier way of taking a pre-processed set of \
    documents, choose a DTM model and get an analysis of the topic's evolution \
    over time.

    Choose a model to work with, create an instance of it by passing the \
    right parameters and then you can instantiate LdaOverTime by passing the \
    previous object.

    :param model: instance of the chosen model.
    :type model: DtmModelInterface

    :return: Nothing
    :rtype: None
    """

    def __init__(self, model: DtmModelInterface) -> None:
        """
        Initialize values and train model.
        """

        # Save parameters
        self.model = model
        self.corpus = self.model.corpus
        self.dates = self.model.dates
        self.dafe_format = self.model.date_format
        self.freq = self.model.freq
        self.n_topics = self.model.n_topics
        self.sep = self.model.sep
        self.workers = self.model.workers

        # Train model
        self.model.train()

        # Get number of time slices
        self.n_timeslices = self.model.n_timeslices

        # Get results from model
        self.results, self.dates, self.weights = self.__get_results()

        # It holds the default name of each topic
        self.topics_names = []

        # Create default topic's name: top 10 words of last time slice
        for topic in range(self.n_topics):
            words = ', '.join(
                    self.model.get_topic_words(
                        topic,
                        self.n_timeslices - 1,
                        10
                    )
            )
            self.topics_names.append(words)

        # Default name is top 10 words
        self.rename_topics(self.topics_names)


    def __get_results(self):
        # extract results that will be used to plot model
        results = self.model.get_results()

        # get weights of each topic over time
        weights = results[list(range(self.n_topics))].copy()

        # get dates
        dates = results['date'].dt.date.values

        # return dates and weights
        return results, dates, weights


    def plot(self,
             title: str,
             legend_title: Optional[str] = None,
             path_to_save: Optional[str] = None,
             rotation: int = 90,
             mode: str = "line",
             display: bool = True,
             date_format: Optional[str] = None):
        """
        Plot the evolution of topics over time.

        To rename topics' names, use method `rename_topics`.

        :param title: title of plot
        :type title: str

        :param legend_title: legend's title
        :type legend_title: str, optional

        :param path_to_save: set it with path to save the graph. Default \
        behaviour does not save the graph.
        :type path_to_save: str, optional

        :param rotation: value in degrees to rotate horizontal labels. Default \
        is 90.
        :type rotation: int, optional

        :param mode: type of plotting. It can be either a simple `line` plot \
        or `stack` plot. Default is `line`.
        :type mode: str, optional

        :param display: set it to False to not display graph. Default \
        behaviour is to display.
        :type display: bool, optional

        :param date_format: date format to be displayed
        :type date_format: str, optional

        :return: Nothing
        :rtype: None
        """
        # Plot lines
        if mode == "line":
            g = sns.lineplot(data=self.weights)
            g.set_xticks(range(len(self.dates)))
            plt.legend(title=legend_title,
                       bbox_to_anchor=(1,1),
                       loc="upper left")

            # date_format was not provided: print with self.date_format
            if date_format is None:
                g.set_xticklabels(labels=self.dates, rotation=rotation)

            # date_format was provided: print with custom labels
            else:
                g.set_xticklabels(
                        labels=[
                            date.strftime(date_format)
                                for date in self.dates
                        ],
                        rotation=rotation
                )

        # Plot stacks
        elif mode == "stack":
            y = [self.weights[col] for col in self.weights.columns]
            sns.set_theme()
            _, ax = plt.subplots()
            ax.stackplot(self.dates, *y)
            ax.legend(labels=self.weights.columns,
                       title=legend_title,
                       bbox_to_anchor=(1,1),
                       loc="upper left")
            plt.xticks(rotation=rotation)

            # Set custom date format if provided
            if date_format is not None:
                ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
                ax.xaxis.set_minor_formatter(mdates.DateFormatter(date_format))

        # Unknown plot
        else:
            raise ValueError(f"There is no option `mode = {mode}`")

        plt.title(title)

        # Path was given: save plot in path
        if isinstance(path_to_save, str):
            plt.savefig(path_to_save)

        # Set to display: display plot
        if display is True:
            plt.show()


    def save(self, file_path: str) -> None:
        """
        Save your current work in the location file_path. You can reload your \
        work later by calling load with the same file_path.

        :param file_path: Location to save your current work.
        :type file_path: str

        :return: Nothing
        :rtype: None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, file_path: str) -> 'LdaOverTime':
        """
        Load your last work.

        :param file_path: Location where you saved your last work.
        :type file_path: str

        :return: Last saved work.
        :rtype: LdaOverTime
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)


    def showvis(self, time_id: int):
        """
        Show the PyLdaVis analysis of your model in a specific time slice. \
        It is useful to evaluate how good your model is.

        *This method is only available inside jupyter notebooks.*

        :param time_id: Position of the time slice from 1 to n_timeslices in \
        chronological order
        :type time_id: int

        :return: Nothing
        :rtype: None
        """
        args = self.model.prepare_args(time_id)

        display = pyLDAvis.prepare(**args)

        return pyLDAvis.display(display)


    def get_topic_words(
        self,
        topic_id: int,
        timeslice: int,
        n: int = 10
    ) -> List[str]:
        """
        Get the top `n` words of from a specific topic in the chosen \
        timeslice.

        :param topic_id: The id of the desired topic.
        :type topic_id: int

        :param timeslice: The position of the desired timeslice in \
        chronological order the first (oldest) time slice is indexed by 1.
        :type timeslice: int

        :param n: This specifies how many words that better describes the \
        topic at a specific time slice should be returned.
        :type n: int

        :return: It returns a list of top n words that best describes the \
        requested topic in a specific time.
        :rtype: list[str]
        """
        return self.model.get_topic_words(topic_id - 1, timeslice, n)


    def get_results(self) -> pd.DataFrame:
        """
        Get the model's result in format of a table.

        In this table, rows represents each time slice.

        For the columns, the `date` column holds the time slices' timestamps \
        and the remaing `n_topics` columns indexed from 1 to `n_topics` \
        holds the proportion of each topic of each time slice.

        You can get each topic's main words by calling `get_topic_words`, \
        e.g. if you want the top 10 words from the topic 3 of this table in \
        the first row, call `get_topic_words(topic_id=3, timeslice=1, n=10)`

        :return: table with results
        :rtype: pandas.DataFrame

        """
        # Get results
        results = self.model.get_results()

        # Change columns to number topics from 1 to n_topics
        results.rename(
            columns={
                i: i + 1
                    for i in range(self.n_topics)
            },
            inplace=True
        )

        # Return table
        return results


    def rename_topics(self, new_names: List[str]):
        """
        Rename topic's names with the list with new names.

        It will rename based on the given order, that is the first name will \
        overwrite the first topic, the second will overwrite second topic, and \
        so on.

        The length should be equal to number of topics, otherwise it will \
        raise ValueError.

        :param new_names: List with new names to overwrite the topics' names
        :type new_names: list[str]

        :return: Nothing
        :rtype: None

        :raises ValueError: when the given list's length does not match with \
        the number of topics.
        """
        # Raise exception if length does not match with number of topics
        if len(new_names) != self.n_topics:
            raise ValueError(
                    f'The given list should have length {self.n_topics}.'
            )

        # Map older names to new names
        renaming = dict(zip(self.weights.columns, new_names))

        # Rename weights' columns
        self.weights.rename(columns=renaming, inplace=True)

