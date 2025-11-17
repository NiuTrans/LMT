<div align="center">
  
# 🌍 Large-scale Multilingual Translation (LMT) 
</div>

 <p align="center">
  <a href="https://github.com/NiuTrans" alt="NiuTrans"><img src="https://img.shields.io/badge/NiuTrans-blue"/></a>
  <a href="http://team.neu.edu.cn/NEUNLPLab/zh_CN/index.htm" alt="NEUNLP"><img src="https://img.shields.io/badge/NEUNLP-blue"/></a>
</p> 


<div align="center">
<p align="center" dir="auto">

•  [📢 News](#-news) 
•  [🤗 Open Resources](#-open-resources)
•  [📄 Contents](#-contents)
</p>

</div>

The LMT aims to advance the frontier of Multilingual Machine Translation (MMT) by building **Inclusive**, **Scalable**, and **High-performance** multilingual translation models.

# 📢 News

- *2025.11.11*: Our **LMT** paper [Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](https://arxiv.org/abs/2511.07003) and corresponding [Models](https://huggingface.co/NiuTrans/LMT-60-8B) are released.

 


# 🤗 Open Resources
We have made the following resources available:

| Resource         | Description                                         | Link                                                      |
|------------------|-----------------------------------------------------|-----------------------------------------------------------|
| LMT-60-*B    | Our high-performance multilingual translation models cover 60 languages and 234 directions. Available in four sizes: 0.6B / 1.7B / 4B / 8B.  | [LMT-60-0.6B](https://huggingface.co/NiuTrans/LMT-60-0.6B) <br> [LMT-60-1.7B](https://huggingface.co/NiuTrans/LMT-60-1.7B) <br> [LMT-60-4B](https://huggingface.co/NiuTrans/LMT-60-4B) <br> [LMT-60-8B](https://huggingface.co/NiuTrans/LMT-60-8B) |
| LMT-60-*B-Base    | Our continued pre-training of Qwen3 on 90B tokens serves as the foundation for large-scale translation adaptation. Available in four sizes: 0.6B / 1.7B / 4B / 8B. | [LMT-60-0.6B-Base](https://huggingface.co/NiuTrans/LMT-60-0.6B-Base) <br> [LMT-60-1.7B-Base](https://huggingface.co/NiuTrans/LMT-60-1.7B-Base) <br> [LMT-60-4B-Base](https://huggingface.co/NiuTrans/LMT-60-4B-Base) <br> [LMT-60-8B-Base](https://huggingface.co/NiuTrans/LMT-60-8B-Base) |
| LMT-60-sft-data   | Our SFT dataset including Flores-200 devset, NTREX-128, SMol, WMT14–23, and IWSLT17–24 test sets, totaling 567K samples.	 | [LMT-60-sft-data](https://huggingface.co/datasets/NiuTrans/LMT-60-sft-data) |
| FLORES-mn_cn   | A new Chinese–Mongolian evaluation set annotated by native speakers to extend the FLORES-200 benchmark.	 | [FLORES-mn_cn](https://huggingface.co/datasets/NiuTrans/FLORES-mn_cn) |

# 📄 Contents 
## Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs

### Introduction
In this project, we take a step toward overcoming the prevailing English-centric bias in MMT. We introduce **LMT**, a suite of **Chinese-English-centric** MMT models trained on **90B** mixed monolingual and bilingual tokens, covering **60 languages across 234 translation directions** and achieving **SOTA performance** among models with similar language coverage.
Our work makes the following main contributions:
- We identify and analyze a previously overlooked issue, **directional degeneration**, in large-scale multilingual SFT with multi-way data and propose a simple yet effective **Strategic Downsampling** method to mitigate it.
- We propose **Parallel Multilingual Prompting (PMP)**, which enhances cross-lingual transfer by incorporating an auxiliary parallel sentence into the instruction.
- We release LMT, a suite of **large-scale Chinese–English-centric** multilingual translation models in four sizes (0.6B/1.7B/4B/8B), providing strong baselines for future MMT research.

### Support Languages

| Resource Tier | Languages |
| :---- | :---- |
| High-resource Languages (13) | Arabic(ar), English(en), Spanish(es), German(de), French(fr), Italian(it), Japanese(ja), Dutch(nl), Polish(pl), Portuguese(pt), Russian(ru), Turkish(tr), Chinese(zh) |
| Medium-resource Languages (18) | Bulgarian(bg), Bengali(bn), Czech(cs), Danish(da), Modern Greek(el), Persian(fa), Finnish(fi), Hindi(hi), Hungarian(hu), Indonesian(id), Korean(ko), Norwegian(nb), Romanian(ro), Slovak(sk), Swedish(sv), Thai(th), Ukrainian(uk), Vietnamese(vi) |
| Low-resouce Languages (29) | Amharic(am), Azerbaijani(az), Tibetan(bo), Modern Hebrew(he), Croatian(hr), Armenian(hy), Icelandic(is), Javanese(jv), Georgian(ka), Kazakh(kk), Central Khmer(km), Kirghiz(ky), Lao(lo), Chinese Mongolian(mn_cn), Marathi(mr), Malay(ms), Burmese(my), Nepali(ne), Pashto(ps), Sinhala(si), Swahili(sw), Tamil(ta), Telugu(te), Tajik(tg), Tagalog(tl), Uighur(ug), Urdu(ur), Uzbek(uz), Yue Chinese(yue) |

### Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "NiuTrans/LMT-60-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Translate the following text from English into Chinese.
English: The concept came from China where plum blossoms were the flower of choice.
Chinese: "
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512, num_beams=5, do_sample=False)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

print("response:", outputs)
```
For more details, please refer to [src/inference.py](./src/inference.py).

## Reference
Email: luoyingfeng_neu@outlook.com

If you find our paper useful for your research, please kindly cite our paper:

```
@misc{luoyf2025lmt,
      title={Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs}, 
      author={Yingfeng Luo, Ziqiang Xu, Yuxuan Ouyang, Murun Yang, Dingyang Lin, Kaiyan Chang, Tong Zheng, Bei Li, Peinan Feng, Quan Du, Tong Xiao, Jingbo Zhu},
      year={2025},
      eprint={2511.07003},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07003}, 
}
```
