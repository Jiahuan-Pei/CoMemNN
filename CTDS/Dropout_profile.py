#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-07-23
"""
import torch
import numpy as np
from CTDS.Config import *

def profile_dropout(profile, profile_dropout_ratio=0.2, keep_attributes=None):
    if profile_dropout_ratio == 0:
        return profile
    else:
        probs = torch.ones_like(profile) * (1 - profile_dropout_ratio)
        # drop the i-th value where v[i]==0
        v = torch.distributions.binomial.Binomial(torch.ones_like(profile), probs).sample()
        # v = np.random.binomial(1, 1-profile_dropout_ratio, size=complete_profile.size())

        # keep the some attribute values, do not drop them
        if keep_attributes is not None:
            for attr in keep_attributes:
                if attr == 'gender': # 0, 1
                    v[0:2] = 1
                elif attr == 'age': # 2, 3, 4
                    v[2:5] = 1
                elif attr == 'dietary': # 5, 6
                    v[5:7] = 1
                elif attr == 'favorite': # 7-21
                    v[7:] = 1

        v.to(profile.device)
        return profile * v


if __name__ == "__main__":
    pass