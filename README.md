# baichuan-ptune-deepPrompt

# baichuan模型 p-tune v2 fine tune

## 增加内容：

```
class PrefixEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2) #[seq layer hid 2]
        self.cfg=config
    def forward(self, prefix: torch.Tensor): # [batch seq]
        batch,seq=prefix.shape
        past_key_values = self.embedding(prefix)
        past_key_values=past_key_values.view(2,self.cfg.num_hidden_layers ,batch,self.cfg.num_attention_heads,seq,self.cfg.hidden_size//self.cfg.num_attention_heads)
        return past_key_values  #[2,layer,batch,head,seq,dim]
        
```


### 在BaiChuanForCausalLM的forward中 做出了修改 ：将past_key_values替换成deep prompt， attention_fn等相应修改
```
##### P-TUNE v2  DEEP PROMPT
config.pre_seq_len=16
self.prefixEncoder=PrefixEncoder(config)
self.prefixToken=  torch.arange(config.pre_seq_len).long()
```
```
prefix_tokens = self.prefixToken.unsqueeze(0).expand(input_ids.shape[0], -1)
past_key_values=self.prefixEncoder(prefix_tokens)
```

## 使用说明

把目录/ptune-deep/模型.py 替换掉huggingface 里的 模型.py

## 参考论文

P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks

## model
https://huggingface.co/baichuan-inc/Baichuan-7B

