import argparse
import logging
import tqdm
import time
import regex
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import reduce
import datetime

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

LANG_TABLE = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ckb": "Central Kurdish",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Modern Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Modern Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kmr": "Northern Kurdish",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "lb": "Luxembourgish",
    "li": "Limburgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn_cn": "Inner Mongolian",
    # "mn": "Halh Mongolian",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "nso": "Northern Sotho",
    "oc": "Occitan",
    "om": "West Central Oromo",
    "or": "Oriya",
    "pa": "Panjabi",
    "pl": "Polish",
    "prs": "Dari",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sm": "Samoan",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Yue Chinese",  
    "zh": "Chinese",
    "zu": "Zulu",
}

def apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer):
    test_dataset = open(args.test_file, encoding='utf8', mode='r').read().strip().split("\n")
    res = []
    for line in test_dataset:
        prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}\n{tgt_fullname}:"
        prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        res.append(text)
    return res

def is_whitespace(string):
    pattern = r'^[\s\p{C}[\x00-\xFF]]+$'
    match = regex.match(pattern, string)
    return match is not None

def clean_pred(pred, remove_special_tokens=[]):
    ## remove special tokens
    for s in remove_special_tokens:
        pred = pred.replace(s, "")
    ## last step: check
    pred = "#" if is_whitespace(pred) else pred
    return pred

def get_special_tokens(tokenizer):
    remove_special_tokens = ["<unk>", "</s>", "<pad>", "\n"]
    if getattr(tokenizer, "pad_token", None):
        remove_special_tokens.append(tokenizer.pad_token)
    if getattr(tokenizer, "eos_token", None):
        remove_special_tokens.append(tokenizer.eos_token)
    if getattr(tokenizer, "bos_token", None):
        remove_special_tokens.append(tokenizer.bos_token)
    if getattr(tokenizer, "unk_token", None):
        remove_special_tokens.append(tokenizer.unk_token)
    return remove_special_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="NiuTrans/LMT-60-8B", required=True)
    parser.add_argument("-t", "--test_file", type=str, default="", required=True)
    parser.add_argument("-l", "--lang_pair", type=str, default='en-zh', required=True)
    parser.add_argument("-s", "--hypo_file", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_batch", type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default='0')
    args = parser.parse_args()
    if not args.hypo_file:
        args.hypo_file = args.test_file + ".predict"
    
    src_lang, tgt_lang = args.lang_pair.split("-")
    src_fullname = LANG_TABLE[src_lang]
    tgt_fullname = LANG_TABLE[tgt_lang]

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    remove_special_tokens = get_special_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    test_dataset = apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer)
    def make_batch(prompts, batch_size):
            batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
            batches_tok = []
            for prompt_batch in batches:
                input_ids = tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False
                ).to(model.device) 
                batches_tok.append(input_ids)
            return batches_tok

    start = time.time()
    results = dict(outputs=[], num_tokens=0)
    prompt_batches = make_batch(test_dataset, batch_size=args.num_batch)
    prompt_batches = tqdm.tqdm(prompt_batches, total=len(prompt_batches))
    for prompts_tokenized in prompt_batches:
        outputs_tokenized = model.generate(
            **prompts_tokenized,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        outputs_tokenized = [ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
        # count and decode gen. tokens 
        num_tokens = sum([ len(t) for t in outputs_tokenized ])
        outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        outputs = list(map(lambda x: clean_pred(x, remove_special_tokens=remove_special_tokens),  outputs))
        results["outputs"].extend(outputs)
        results["num_tokens"] += num_tokens

    timediff = time.time() - start
    num_tokens = results["num_tokens"]
    preds = results["outputs"]
    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
    with open(args.hypo_file, mode='w') as fout:
        fout.write("\n".join(preds) + '\n')


if __name__ == "__main__":
    main()
