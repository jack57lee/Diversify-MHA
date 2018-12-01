#!/usr/bin/env python
#coding: utf8

# from __future__ import unicode_literals, division

import argparse
import traceback
import pprint
import logging
import time
import sys
import os


import BaseHTTPServer
from subprocess import Popen, PIPE
import urllib
from urllib import unquote_plus
import re
import xmlrpclib

logger = logging.getLogger(__name__)


# -*- coding:utf-8 -*-  
import datetime
import commands

import json
import numpy as np
import os
from numpy import array


def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  for i, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      # Sum across different attention values for each token.
      att = att[:, :, :max_length, :max_length]
      row_sums = np.sum(att, axis=2)
      # Normalize
      att /= row_sums[:, :, np.newaxis]
    att_mat[i] = att
  return att_mat


def _get_attention(inp_text, out_text, enc_atts, dec_atts, encdec_atts):
  """Compute representation of the attention ready for the d3 visualization.

  Args:
    inp_text: list of strings, words to be displayed on the left of the vis
    out_text: list of strings, words to be displayed on the right of the vis
    enc_atts: numpy array, encoder self-attentions
        [num_layers, batch_size, num_heads, enc_length, enc_length]
    dec_atts: numpy array, decoder self-attentions
        [num_layers, batch_size, num_heads, dec_length, dec_length]
    encdec_atts: numpy array, encoder-decoder attentions
        [num_layers, batch_size, num_heads, enc_length, dec_length]

  Returns:
    Dictionary of attention representations with the structure:
    {
      'all': Representations for showing all attentions at the same time.
      'inp_inp': Representations for showing encoder self-attentions
      'inp_out': Representations for showing encoder-decoder attentions
      'out_out': Representations for showing decoder self-attentions
    }
    and each sub-dictionary has structure:
    {
      'att': list of inter attentions matrices, one for each attention head
      'top_text': list of strings, words to be displayed on the left of the vis
      'bot_text': list of strings, words to be displayed on the right of the vis
    }
  """
  def get_full_attention(layer):
    """Get the full input+output - input+output attentions."""
    enc_att = enc_atts[layer][0]
    dec_att = dec_atts[layer][0]
    encdec_att = encdec_atts[layer][0]
    enc_att = np.transpose(enc_att, [0, 2, 1])
    dec_att = np.transpose(dec_att, [0, 2, 1])
    encdec_att = np.transpose(encdec_att, [0, 2, 1])
    # [heads, query_length, memory_length]
    enc_length = enc_att.shape[1]
    dec_length = dec_att.shape[1]
    num_heads = enc_att.shape[0]
    first = np.concatenate([enc_att, encdec_att], axis=2)
    second = np.concatenate(
        [np.zeros((num_heads, dec_length, enc_length)), dec_att], axis=2)
    full_att = np.concatenate([first, second], axis=1)
    return [ha.T.tolist() for ha in full_att]

  def get_inp_inp_attention(layer):
    att = np.transpose(enc_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_inp_attention(layer):
    att = np.transpose(encdec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_out_attention(layer):
    att = np.transpose(dec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_attentions(get_attention_fn):
    num_layers = len(enc_atts)
    attentions = []
    for i in range(num_layers):
      attentions.append(get_attention_fn(i))

    return attentions

  attentions = {
      'all': {
          'att': get_attentions(get_full_attention),
          'top_text': inp_text + out_text,
          'bot_text': inp_text + out_text,
      },
      'inp_inp': {
          'att': get_attentions(get_inp_inp_attention),
          'top_text': inp_text,
          'bot_text': inp_text,
      },
      'inp_out': {
          'att': get_attentions(get_out_inp_attention),
          'top_text': inp_text,
          'bot_text': out_text,
      },
      'out_out': {
          'att': get_attentions(get_out_out_attention),
          'top_text': out_text,
          'bot_text': out_text,
      },
  }

  return attentions
  
def show(inp_text, out_text, enc_atts, dec_atts, encdec_atts):
  enc_att, dec_att, encdec_att = (resize(enc_atts),
                                  resize(dec_atts), resize(encdec_atts))
  attention = _get_attention(
      inp_text, out_text, enc_att, dec_att, encdec_att)
	  
  #return json.dumps(attention)
  return attention



class MTReqHandler(BaseHTTPServer.BaseHTTPRequestHandler):		
    def do_POST(self):
        path = self.path 
	#print path
        #获取post提交的数据  
        datas = self.rfile.read(int(self.headers['content-length']))
        #print datas	
        #datas = urllib.unquote(path).decode("utf-8", 'ignore')  
        datas = datas.split('&')
        self.send_response(200)  
        self.send_header("Content-type","text/html")  
        #self.send_header("test","This is test!")  
        self.end_headers()  
        inp_text = datas[0].split('=')[1].split()+["<EOS>"]
	sl = len(inp_text)
        out_text = ["<BOS>"]+datas[1].split('=')[1].split()+["<EOS>"]
	tl = len(out_text)
        enc_atts = eval(datas[2].split('=')[1])[:,None,:,:sl,:sl]
        dec_atts = eval(datas[4].split('=')[1])[:,None,:,:tl,:tl]
        encdec_atts = eval(datas[3].split('=')[1])[:,None,:,:tl,:sl]
        att = show(inp_text, out_text, enc_atts, dec_atts, encdec_atts)
        self.wfile.write(att)  	

def parse_args():
    parser = argparse.ArgumentParser(
            "Visualization Server")
    parser.add_argument("--port", help="Port to use", type=int, default=5000)

    return parser.parse_args()

def main():
    args = parse_args()

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, MTReqHandler)

    print 'Visualization Server starting..'
    httpd.serve_forever()


if __name__ == "__main__":
    main()
