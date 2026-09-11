"""Single source of truth for the data-contract schema version (ARCHITECTURE.md §4).

Bump this whenever a schema in this package changes shape in a way that
breaks previously-serialized artifacts.
"""

SCHEMA_VERSION = "0.1"
