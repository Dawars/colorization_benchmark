---
title: 'Deepremaster: Multi reference'
layout: default
tag: deepremaster
category: multi_reference
last_modified_at: "2024-03-20 21:04:22.842727"
---
# Multi reference
## Deepremaster

Paper: <{{ "https://github.com/satoshiiizuka/siggraphasia2019_remastering" | uri_escape  }}>{:target="_blank"}

This model is originally designed for film colorization.
To run this benchmark the input image is duplicated 5 times.
The reference images are supposed to be colored frames chosen from the movies.

This means that significant differences in the reference images cannot be used, as illustrated below.


An interesting finding is that certain objects are colored even when they don't appear on the refernce images, as long as those colors are present in the reference images.
This suggests that instead of semantic to semantic matching between gray and reference image, semantic to color correspondence is learned (at least partially).
For example, the sky is colored blue and the leaves green.
The semantic matching takes place in feature space where the spatial information is degraded.
See noise test vs gray test.


|  Task | Image #1 | Image #2 | Image #3 | Reference |
|  ----- | ----- | ----- | ----- | ----- |
| Recolor source | ![]({{ "benchmark/multi_reference/deepremaster/recolor_source/0/fortepan_183722_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/recolor_source/0/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/recolor_source/1/fortepan_250610_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/recolor_source/1/fortepan_250610_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/recolor_source/2/fortepan_183723_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/recolor_source/2/fortepan_183723_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/recolor_source/3/fortepan_251236_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/recolor_source/3/fortepan_251236_chromaticity.png" | relative_url }}){: width="200px"} |
|  Task | Image #1 | Image #2 | Image #3 | Reference |
|  ----- | ----- | ----- | ----- | ----- |
| Full correspondence | ![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/0/fortepan_201867_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/0/fortepan_201867_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/1/fortepan_229825_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/1/fortepan_229825_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/2/fortepan_102400_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/full_correspondence/2/fortepan_102400_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Partial source | ![]({{ "benchmark/multi_reference/deepremaster/partial_source/0/fortepan_18476_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/partial_source/0/fortepan_18476_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/partial_source/1/fortepan_79821_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/partial_source/1/fortepan_79821_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/partial_source/2/fortepan_67270_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/partial_source/2/fortepan_67270_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Semantic correspondence strong | ![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/0/fortepan_251148_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/0/fortepan_251148_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/1/fortepan_97196_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/1/fortepan_97196_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/2/fortepan_97191_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_strong/2/fortepan_97191_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Semantic correspondence weak | ![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/0/fortepan_148611_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/0/fortepan_148611_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/1/fortepan_84203_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/1/fortepan_84203_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/2/fortepan_84203_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/semantic_correspondence_weak/2/fortepan_84203_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Distractors | ![]({{ "benchmark/multi_reference/deepremaster/distractors/0/fortepan_18098_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/distractors/0/fortepan_18098_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/distractors/1/fortepan_276876_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/distractors/1/fortepan_276876_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/deepremaster/distractors/2/fortepan_40115_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/deepremaster/distractors/2/fortepan_40115_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |

### Additional Information

{% include_relative footer.md %}

### Other Categories:

{% for p in site.pages %}
{% if p.tag == "deepremaster" and p.url != page.url %}
- [{{p.title}}]({{p.url | relative_url}})
{% endif %}
{% endfor %}
