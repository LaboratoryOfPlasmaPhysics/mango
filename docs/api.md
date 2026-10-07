# API reference

```{eval-rst}
.. autofunction:: space_mango.get_data

.. autofunction:: space_mango.regions

.. autofunction:: space_mango.columns

.. autofunction:: space_mango.filters

.. autofunction:: space_mango.describe

.. autofunction:: space_mango.spacecraft

.. autofunction:: space_mango.count

.. autofunction:: space_mango.search

.. autofunction:: space_mango.timeline

.. autofunction:: space_mango.cite

.. autofunction:: space_mango.dataset_info

.. py:data:: space_mango.cache

   Handle on the default client's on-disk cache: ``mango.cache.info()`` returns its location,
   number of files, size and size cap; ``mango.cache.clear()`` empties it.

.. autoclass:: space_mango.MangoClient
   :members:

.. autoclass:: space_mango.MangoResult
   :members:

.. autoclass:: space_mango._regions_generated.MagnetosphereAPI
   :members: get_data, count, describe, spacecraft

.. autoclass:: space_mango._regions_generated.MagnetosheathAPI
   :members: get_data, count, describe, spacecraft

.. autoclass:: space_mango._regions_generated.SolarWindAPI
   :members: get_data, count, describe, spacecraft

.. automodule:: space_mango.errors
   :members: MangoError, UnknownRegionError, UnknownSpacecraftError, UnknownColumnError, MangoFilterError, TimeParseError, ServerError, CacheMissError
```
