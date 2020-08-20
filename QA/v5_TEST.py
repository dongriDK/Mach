import requests
import random
import json
import v5_response_to_file
import time
from datetime import datetime
import logging


class v5_TEST:
    def __init__(self, api_type, path, api_key, api_url):
        self._api_type = api_type
        self._path = path
        self._api_key = api_key
        self._api_url = api_url
        self._list = self.read_file()
        self._type = ['hash','ip','hostname','url']

    def v5_request(self):
        rand_value = random.choice(self._list)[:-1]
        params = {'api_key': self._api_key, self._type[self._api_type]: rand_value}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.request_edit_header(rand_value,params,headers)

    def read_file(self):
        with open(self._path, "r") as f:
            tmp_list = f.readlines()
        return tmp_list

    def request_edit_header(self, rand_value, params, headers):
        process_start_time = datetime.now()
        try:
            response = requests.post(self._api_url, data=params, headers=headers)
        except:
            logging.error("requests error : " + response.headers)

        process_time = round(self.time_millis(process_start_time))
        para = {"type":self._type[self._api_type], "value":rand_value, "processtime":process_time}
        try:
            parameter = json.loads(json.dumps(para))
        except:
            logging.error("para type is not a dict")

        calc_response = v5_response_to_file.response_to_file(response.json(), response.headers, response.status_code, process_start_time, parameter)
        calc_response.edit_header()

    def time_millis(self, start_time):
       dt = datetime.now() - start_time
       ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000
       return ms
