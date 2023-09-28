# -*- coding: utf-8 -*-
class PertNet(nn.Module):
    def __init__(self,init_vector):
        super(PertNet, self).__init__()
        #Define vector to optimize
        self.d = nn.Parameter(init_vector)