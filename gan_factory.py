from gan import *
from gan_cls import *

class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'vanilla_gan':
            return gan.generator()
        
    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'vanilla_gan':
            return gan.discriminator()
    
