class NotImplemented(Exception):
    """
    NotImplemented is an error that occurs when a method is defined in an
    interface but not implemented.
    """

    def __init__(self, method_name, *args):
        """
        This initializes the NotImplemented exception and it takes the
        method_name that raised the exception.

        :param method_name: The method's name that was not implemented.
        :type method_name: str

        :return: nothing
        :rtype: None
        """
        self.method_name = method_name
        super().__init__(args)


    def __str__(self):
        """
        This method overwrites the exception's string to a custom message that
        informs the not implemented method's name.

        :return: Message informing the missing implementation of method.
        :rtype: str
        """
        return f'Method "{self.method_name}" was not implemented!'
