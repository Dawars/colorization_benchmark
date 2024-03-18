---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

![]({{ "assets/teaser_left.png" | relative_url}})

Unconditional Image Colorization
{% for p in site.pages %}
{% if p.category == "unconditional" %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endif %}
{% endfor %}

Example-based Image Colorization with Single Reference Images
{% for p in site.pages %}
{% if p.category == "single_reference" %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endif %}
{% endfor %}

Example-based Image Colorization with Multiple Reference Images
{% for p in site.pages %}
{% if p.category == "multi_reference" %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endif %}
{% endfor %}


## Copyright information

The photos used in these benchmarks come from [Fortepan](https://fortepan.hu/en) with the [CC BY-SA 3.0 DEED](https://creativecommons.org/licenses/by-sa/3.0/deed.en) licence.

We thank them and the donors for their hard work and making the photos freely available.