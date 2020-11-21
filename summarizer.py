


def make_summarizer(method):
    return summarizer(method)


class summarizer():
    def __init__(self,method):
        self.method = method

    def sum_text(self, x):
        return x