import boto3
import re

import os
import numpy as np
import pandas as pd

from sagemaker import get_execution_role
import sagemaker as sage

role = 'awais'

sess = sage.Session()

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/mlfw_test'.format(account, region)

print(image)

clf = sage.estimator.Estimator(image,
                               role, 1, 'ml.c4.2xlarge',
                               output_path="s3://vimla-data-modelling/mlfw",
                               sagemaker_session=sess)


clf.fit()
