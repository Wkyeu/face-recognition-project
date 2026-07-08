#app using streamlit (reruns the file everytime interaction is detected)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
from datetime import datetime 
import cv2
import numpy as np #to convert photo from streamlit into a openCV format
import pandas as pd #displa/sort csv as a proper table
import streamlit as st 
from deepface import DeepFace 

