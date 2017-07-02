#!/usr/bin/env python

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import json
import os
import korbit

from flask import Flask
from flask import request
from flask import make_response

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processRequest(req):
    data = korbit.detailed_ticker() 
    res = makeWebhookResult(data)
    return res

# TODO
def makeQuery(req):
    result = req.get("result")
    parameters = result.get("parameters")
    last = parameters.get("last")

def makeWebhookResult(data):
    speech = "Price of the last filled order : " + data["last"] + "\n"\
            "Best bid price : " + data["bid"] + "\n"\
            "Best ask price : " + data["ask"] + "\n"\
            "Lowest price within the last 24 hours : " + data["low"] + "\n"\
            "Highest price within the last 24 hours : " + data["high"] + "\n"\
            "Transaction volume within the last 24 hours : " + data["volume"] + "\n"
    
    print("Response:")
    print(speech)

    return {
        "speech": speech,
        "displayText": speech,
        "source": "apiai-korbit-webhook"
    }


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8090))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
