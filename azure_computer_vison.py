from array import array
import os
from PIL import Image
import sys
import time

import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


# https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management
KEY = st.secrets.AzureApiKey.key
ENDPOINT = st.secrets.AzureApiKey.endpoint

computervision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))


def get_tags(filepath):
    local_image = open(filepath, 'rb')

    tags_result = computervision_client.tag_image_in_stream(local_image)
    tags = tags_result.tags

    tags_name = []
    for tag in tags:
        tags_name.append(tag.name)
    return tags_name


def detect_objects(filepath):
    local_image = open(filepath, 'rb')
    detect_objects_results = computervision_client.detect_objects_in_stream(local_image)
    objects = detect_objects_results.objects
    return objects
