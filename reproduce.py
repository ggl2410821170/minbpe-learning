"""Compare basic/regex BPE on a small synthetic multilingual corpus."""
import json
import tempfile
from pathlib import Path
from minbpe import BasicTokenizer, RegexTokenizer

TRAIN = '\n'.join([
    'Machine learning models learn patterns from training data.',
    'Medical imaging research compares model predictions and labels.',
    'A tokenizer converts text into tokens and tokens into text.',
    '人工智能帮助研究者分析数据，模型需要训练与验证。',
    '医学图像分类是机器学习的应用，实验需要记录结果。',
    'Tokenizer实验：Python 3.12，accuracy=0.95，hello world! 🙂',
]*12)
HELD_OUT = [
    '模型预测结果需要验证，医学研究重视实验。',
    'Machine learning research needs careful evaluation.',
    'AI与医学：score=0.87; next step? 🧠',
    '', '?', '你好，世界！🙂',
]


def main():
    results = []
    for cls in [BasicTokenizer, RegexTokenizer]:
        for size in [256, 280, 320]:
            tok = cls()
            tok.train(TRAIN, size)
            entries = []
            for text in HELD_OUT:
                ids = tok.encode(text)
                assert tok.decode(ids) == text
                entries.append({'text':text,'utf8_bytes':len(text.encode('utf-8')),'tokens':len(ids)})
            with tempfile.TemporaryDirectory() as temp:
                prefix = str(Path(temp)/'tokenizer')
                tok.save(prefix)
                restored = cls()
                restored.load(prefix+'.model')
                for text in HELD_OUT:
                    assert restored.encode(text)==tok.encode(text)
            total_bytes = sum(x['utf8_bytes'] for x in entries)
            total_tokens = sum(x['tokens'] for x in entries)
            results.append({'tokenizer':cls.__name__,'vocab_size':size,
                'round_trip_checks':len(HELD_OUT),'save_load_checks':len(HELD_OUT),
                'held_out_utf8_bytes_per_token':total_bytes/total_tokens,'samples':entries})
    result = {'corpus':'Synthetic educational corpus; no patient data.',
              'limitation':'Tiny illustrative benchmark; not representative of medical NLP performance.',
              'runs':results}
    (Path(__file__).parent/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'runs':len(results),'checks':sum(x['round_trip_checks']+x['save_load_checks'] for x in results)}))


if __name__ == '__main__': main()
