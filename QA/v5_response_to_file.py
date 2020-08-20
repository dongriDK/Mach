import json
import random
from datetime import datetime

class response_to_file:
    def __init__(self, body, header, status, time, parameter):
        self._body = body
        self._header = header
        self._status = str(status)
        self._time = str(time.strftime('%H.%M.%S.%f'))
        self._parameter = parameter

    def edit_header(self):
        old_header = json.loads(json.dumps(dict(self._header)))
        old_header["Status"]="HTTP/1.1 "+self._status
        append_parameter = {"parameter":self._parameter}

        savejson_body = {"body":self._body}
        savejson = old_header
        savejson.update(append_parameter)

        self.json_to_file(savejson)
        self.json_to_file_body(savejson_body)

    def json_to_file(self, json_result):
        # with open("./v5_TEST_RESULT/"+self._parameter["type"]+"_"+self._time+"_"+self._status+"_header"+'.txt', "w") as f:
        with open("./v5_TEST_RESULT/"+self._time+"_"+self._parameter["type"]+"_"+self._status+"_header"+'.txt', "w") as f:
            f.write(json.dumps(json_result))

    def json_to_file_body(self, json_result):
        # with open("./v5_TEST_RESULT/"+self._parameter["type"]+"_"+self._time+"_"+self._status+"_body"+'.txt', "w") as f:
        with open("./v5_TEST_RESULT/"+self._time+"_"+self._parameter["type"]+"_"+self._status+"_body"+'.txt', "w") as f:
            f.write(json.dumps(json_result))
