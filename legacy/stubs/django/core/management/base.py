"""Minimal stand-in for Django's BaseCommand so ids.py can run outside Django."""
class CommandError(Exception):
    pass

class BaseCommand:
    def __init__(self, *a, **k):
        pass
