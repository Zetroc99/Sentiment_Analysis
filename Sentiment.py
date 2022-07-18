import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report