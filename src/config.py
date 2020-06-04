#!/usr/bin/env python3
import yaml

class Config:
    def __init__(self, path):
        self._c = {}
        self.path = path
        self.reload()

    def reload(self, method=None):
        with open(self.path, 'r') as f:
            obj = yaml.safe_load(f)
            self.method = method if method else obj['method']
            self._c.update(obj['general'])
            self._c.update(obj['cluster'])
            self._c.update(obj['methods'][self.method])

    def __getattr__(self, key):
        try:
            return self._c[key]
        except KeyError:
            raise AttributeError(key)

    def __str__(self):
        return '\n'.join(['{}:  {}'.format(k,v) for k,v in self._c.items()])

if __name__=='__main__':
    c = Config('config.yaml')
    print(c)
