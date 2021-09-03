#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:15:08 2020

@author: Asep Fajar Firmansyah
"""
import argparse
from data_loader import gen_literal

def main(ds_name):
    gen_literal(ds_name)

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='KGSUMM: Preparing data for rotate embeddings')
      parser.add_argument("--ds_name", type=str, default="dbpedia", help="use dbpedia or lmdb")
      args = parser.parse_args()
      main(args.ds_name)
