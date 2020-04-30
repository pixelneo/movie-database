#!/usr/bin/env python3
import configparser
import json

class ConfigSection:
    def __init__(self, section):
        self._s = section

    def __getattr__(self, key):
        try:
            return self._s[key]
        except KeyError:
            raise AttributeError(key)

class Config:
    def __init__(self):
        self._c = {}
        with open(path, 'r') as f: 
            self._c.update(json.load(f))

    def __getattr__(self, key):
        try:
            return self._c['key']
        except KeyError:
            raise AttributeError(key)

    # def __str__(self):
        # return '\n'.join(['{}:  {}'.format(k,v) for k,v in self._data.items()])

if __name__=='__main__':
    c = Config.from_file('config.ini')
    # print(c.config['main']['lemma'])
    # print(c.config.sections())
    # print(c.config.options('main'))
    # print(c.config['main'])
    print(type(c.main.a))
    print(type(c.main.b))
    print(type(c.main.c))
