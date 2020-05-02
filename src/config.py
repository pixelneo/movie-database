#!/usr/bin/env python3
import json

class Config:
    def __init__(self, path):
        self._c = {}
        with open(path, 'r') as f: 
            obj = json.load(f)
            self.method = obj['method']
            self._c.update(obj['general'])
            self._c.update(obj['methods'][self.method])

    def __getattr__(self, key):
        try:
            return self._c[key]
        except KeyError:
            raise AttributeError(key)

    def __str__(self):
        return '\n'.join(['{}:  {}'.format(k,v) for k,v in self._c.items()])

if __name__=='__main__':
    c = Config('config.json')
    print(c)
