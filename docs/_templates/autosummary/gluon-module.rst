{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% if attributes %}
.. rubric:: {{ _('Module Attributes') }}

.. autosummary::
   :nosignatures:
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: {{ _('Exceptions') }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:
   :template: autosummary/gluon-module.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
