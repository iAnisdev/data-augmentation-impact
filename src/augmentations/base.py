class Augmentation:
    def __init__(self, name):
        self.name = name

    def __call__(self, img):
        raise NotImplementedError
