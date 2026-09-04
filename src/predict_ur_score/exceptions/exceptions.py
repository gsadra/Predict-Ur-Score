'''Custom exceptions'''


class PredictorBaseException(Exception):
    '''Base exception for all related errors'''
    ...

class PredictorInvalidInput(PredictorBaseException):
    '''Raises when input is invalid'''
    ...

class PredictorInvalidModule(PredictorBaseException):
    '''Raises when an invalid module is being called'''
    ...

class PredictorInvalidData(PredictorBaseException):
    '''Raises when an Invalid date is being used'''
    ...

class PredictorFileNotFound(FileNotFoundError):
    '''Raises when a file is not found'''
    ...

class PredictorInvalidTopic(PredictorBaseException):
    '''Raises when an invalid topic is being used'''
    ...