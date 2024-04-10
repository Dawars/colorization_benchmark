---
title: 'Bistnet: Multi reference'
layout: default
tag: BiSTNet
category: multi_reference
last_modified_at: "2024-04-10 10:47:18.744310"
---
# Multi reference
## Bistnet

Paper: <{{ "https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION" | uri_escape  }}>{:target="_blank"}

This method is originally designed for video colorization and is limited to 2 reference images.
The model is changed so that the input is 2 identical frames. The flow map is set to zeros.
TODO: the similarity fusion is done using argmax for an arbitrary number of reference images.

|  Task | Image #1 | Image #2 | Image #3 | Reference |
|  ----- | ----- | ----- | ----- | ----- |
| Recolor source | ![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/0/fortepan_183722_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/0/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/1/fortepan_250610_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/1/fortepan_250610_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/2/fortepan_183723_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/2/fortepan_183723_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/3/fortepan_251236_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/recolor_source/3/fortepan_251236_chromaticity.png" | relative_url }}){: width="200px"} |
|  Task | Image #1 | Image #2 | Image #3 | Reference |
|  ----- | ----- | ----- | ----- | ----- |
| Full correspondence | ![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/0/fortepan_201867_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/0/fortepan_201867_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/1/fortepan_229825_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/1/fortepan_229825_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/2/fortepan_102400_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/full_correspondence/2/fortepan_102400_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Partial source | ![]({{ "benchmark/multi_reference/BiSTNet/partial_source/0/fortepan_18476_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/partial_source/0/fortepan_18476_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/partial_source/1/fortepan_79821_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/partial_source/1/fortepan_79821_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/partial_source/2/fortepan_67270_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/partial_source/2/fortepan_67270_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Semantic correspondence strong | ![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/0/fortepan_251148_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/0/fortepan_251148_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/1/fortepan_97196_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/1/fortepan_97196_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/2/fortepan_97191_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_strong/2/fortepan_97191_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Semantic correspondence weak | ![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/0/fortepan_148611_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/0/fortepan_148611_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/1/fortepan_84203_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/1/fortepan_84203_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/2/fortepan_84203_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/semantic_correspondence_weak/2/fortepan_84203_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |
| Distractors | ![]({{ "benchmark/multi_reference/BiSTNet/distractors/0/fortepan_18098_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/distractors/0/fortepan_18098_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/distractors/1/fortepan_276876_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/distractors/1/fortepan_276876_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/multi_reference/BiSTNet/distractors/2/fortepan_40115_color.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/multi_reference/BiSTNet/distractors/2/fortepan_40115_chromaticity.png" | relative_url }}){: width="200px"} |![]({{ "benchmark/input/fortepan_183722.jpg" | relative_url }}){: width="200px"}![]({{ "benchmark/input/fortepan_183722_chromaticity.png" | relative_url }}){: width="200px"} |

### Additional Information

{% include_relative footer.md %}

### Other Categories:

{% for p in site.pages %}
{% if p.tag == "BiSTNet" and p.url != page.url %}
- [{{p.title}}]({{p.url | relative_url}})
{% endif %}
{% endfor %}
